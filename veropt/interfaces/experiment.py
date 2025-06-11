from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Optional, Literal, Union
import time
import json
import os
from veropt.interfaces.simulation import SimulationResult, SimulationRunner
from veropt.interfaces.batch_manager import BatchManager, BatchManagerFactory, ExperimentMode
from veropt.interfaces.result_processing import ResultProcessor
from veropt.mock_optimiser import MockOptimiser, OptimiserObject

SR = TypeVar("SR", bound=SimulationRunner)
ConfigType = TypeVar("ConfigType", bound=BaseModel)

# TODO: Aster, how to handle nan inputs? 
# TODO: Aster, do we implement an option to minimise or maximise in the experiment 
#       (and maximise in VerOpt core by default)?
# TODO: Aster, should VerOpt core be able to read in pre-simulated initial points?
# TODO: Aster, how to ensure that the objectives passed on to the optimiser
#       are in the correct order?    
# TODO: Should the console output be saved when experiment is finished or stopped?
# TODO: Aster, here is the list of what experiment wants from the optimiser:
#       - At minimum, what info do I have to pass to the optimiser to initialise it?
#       - Default hyperparameter options for default models
#       - Should Experiment take in the optimiser object in order to change the hyperparameters easily?
#       - How to run a single optimization step?
#       - How to access objective functions vals and coords in order to sanity check?
#       - If possible, a log of what optimiser does per optimization step


class Point(BaseModel):
    parameters: Dict[str, float]
    objective_value: Union[float, List[float]]
    state: Optional[str] = None
    processing_method: Optional[str] = None
    job_id: Optional[int] = None


# TODO: Should include timestamps?
class ExperimentalState(BaseModel):
    experiment_name: str
    experiment_directory: str
    points: Dict[int, Point] = {}
    next_point: int = 0

    def update(
        self, 
        new_points: Dict[int, Point],
        next_point: int,
        save: bool = False,
        path_to_json: Optional[str] = None
    ) -> None:
        
        for index, point in new_points.items():
            self.points[index] = point

        self.next_point = next_point

        if save: self.save_to_json(
            path=path_to_json
        )

    def save_to_json(
        self, 
        path: str, 
        **json_kwargs: dict
    ) -> None:
        """Serialize this state to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            fp.write(self.json(**json_kwargs))

    @classmethod
    def load_from_json(
        cls, 
        path: str
    ) -> "ExperimentalState":
        """Load state from JSON file; returns a fresh state if not found."""
        if not os.path.exists(path):
            return cls()
        
        return cls.parse_file(path)


class ExperimentConfig(BaseModel):
    experiment_name: str
    parameter_names: List[str]
    parameter_bounds: Dict[str,List[float]]
    path_to_experiment: str
    experiment_mode: str
    experiment_directory_name: Optional[str] = None
    run_script_filename: str
    run_script_directory: Optional[str] = None
    results_directory_name: Optional[str] = None
    output_filename: str

    @classmethod
    def load_from_json(
        cls, 
        path: str
    ) -> "ExperimentConfig":
        # TODO: Does BaseModel already check if the path exists? 
        #       Maybe this is redundant.
        if not os.path.exists(path):
            return cls()
        
        return cls.parse_file(path)


class Experiment:
    def __init__(
        self, 
        simulation_runner: SimulationRunner,
        result_processor: ResultProcessor,
        experiment_config: Union[str, ExperimentConfig],
        state: Optional[Union[str, ExperimentalState]] = None,
        batch_manager: Optional[BatchManager] = None
    ) -> None:
        
        if isinstance(experiment_config, str):
            self.experiment_config = ExperimentConfig.load_from_json(experiment_config)
        else:
            self.experiment_config = experiment_config

        # TODO: This should be a part of the pathing class
        if self.experiment_config.experiment_directory_name is not None:
            self.experiment_directory = os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_directory_name)
        else:
            self.experiment_directory = os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_name)
    
        if state is None:
            self.state = ExperimentalState(
                experiment_name = self.experiment_config.experiment_name,
                experiment_directory = self.experiment_directory,
                points = {},
                next_point = 0
            )

        elif isinstance(state, str):
            self.state = ExperimentalState.load_from_json(state)

        else: 
            self.state = state

        self.simulation_runner = simulation_runner

        self.result_processor = result_processor

        self.batch_manager = batch_manager

        self.experimental_set_up_initialised = False

    def initialise_directory_structure(
            self
    ) -> None:
        ...

    def initialise_optimiser(
            self
    ) -> None:
        ...

    def initialise_batch_manager(
            self
    ) -> None:
        
        self.batch_manager_config = BatchManagerFactory.make_batch_manager_config(
            experiment_mode=self.experiment_config.experiment_mode
        )
        
        self.batch_manager = BatchManagerFactory.make_batch_manager(
            experiment_mode=self.experiment_config.experiment_mode,
            simulation_runner=self.simulation_runner,
            config=self.batch_manager_config
        )

    # TODO: is this redundant?
    def _check_initialization(
            self
    ) -> None:
        assert isinstance(self.simulation_runner, SimulationRunner)
        assert isinstance(self.batch_manager, BatchManager)
        assert isinstance(self.result_processor, ResultProcessor)

    def get_parameters_from_optimiser(
        self
    ) -> List[Dict[str,float]]:
        ...

    def send_objectives_to_optimiser(
            self,
            objectives: List[float]
    ) -> None:
    # TODO: Is running an opt step correct to do here?
    #       Or should VerOpt automatically run an opt step when receiving objectives
    #       with a loader method?
        ...

    def _sanity_check(
            self,
            list_of_parameters: List[Dict[str,float]],
            results: List[SimulationResult],
            objectives: List[float]
    ) -> None:
        """
        Check if simulations ran with correct params;
        Whether params and objectives match correctly;
        ...
        """
        ...

    def initialise_experimental_set_up(
            self
    ) -> None:
        
        self.initialise_directory_structure()
        self.initialise_optimiser()
        
        if self.batch_manager is None:
            self.initialise_batch_manager()

        self._check_initialization()

        self.experimental_set_up_initialised = True
        
    def run_optimization_step(
            self
    ) -> None:
            
            assert self.experimental_set_up_initialised == True

            list_of_parameters = self.get_parameters_from_optimiser()
            results = self.batch_manager.run_batch(list_of_parameters)
            objectives = self.result_processor.process(results)
            self._sanity_check(list_of_parameters, results, objectives)
            self.state.update(list_of_parameters, results, objectives)
            self.state.save_to_json(self.path)  # should saving to json be a part of state.update?
            self.send_objectives_to_optimiser(objectives)

    def run_optimization_experiment(
            self
    ) -> None:
        
        # initialization
        if not self.experimental_set_up_initialised:
            self.initialise_experimental_set_up()
        
        # run
        for i in self.n_iterations:
            self.run_optimization_step()

    def restart_optimization_experiment(
            self
    ) -> None:
        ...
