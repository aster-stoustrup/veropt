from pydantic import BaseModel, Field
from typing import Any, Dict, List, TypeVar, Optional
import time
import json
import os
from veropt.interfaces.simulation import SimulationResult, SimulationRunner
from veropt.interfaces.batch_manager import BatchManager, BatchManagerFactory
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

class ExperimentalState(BaseModel):
    history: List[Dict[str, Any]] = Field(default_factory=list)
    last_update: float = 0.0

    def update(
        self, 
        params: Dict[str, Any], 
        objective: float, 
        metadata: Dict[str, Any]
    ) -> None:
        """Add a new record and bump the timestamp."""
        record = {
            "params": params,
            "objective": objective,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        self.history.append(record)
        self.last_update = record["timestamp"]

    def save_to_json(
        self, 
        path: str, 
        **json_kwargs
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


class Experiment:
    def __init__(
        self, 
        simulation_runner: SimulationRunner,
        result_processor: ResultProcessor,
        experiment_config: ConfigType,
        state: Optional[ExperimentalState] = None,
        batch_manager: Optional[BatchManager] = None
    ) -> None:
        
        self.experiment_config = experiment_config
        self.state = ExperimentalState() if state is None else state
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
    ) -> List[dict]:
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
            list_of_parameters: List[dict],
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
            self.state.save_to_json(self.path)
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
