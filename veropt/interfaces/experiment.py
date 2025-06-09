from pydantic import BaseModel, Field
from typing import Any, Dict, List, TypeVar, Optional
import time
import json
import os
from veropt.interfaces.simulation import SimulationResult, SimulationRunner
from veropt.interfaces.batch_manager import BatchManager, BatchManagerFactory
from veropt.interfaces.result_processing import ResultProcessor

SR = TypeVar("SR", bound=SimulationRunner)
ConfigType = TypeVar("ConfigType", bound=BaseModel)

# TODO: Aster, how to handle nan inputs? 
# TODO: Aster, do we implement an option to minimize or maximize in the experiment 
#       (and maximize in VerOpt core by default)?
# TODO: Aster, should VerOpt core be able to read in pre-simulated initial points?
# TODO: Aster, how to ensure that the objectives passed on to the optimizer
#       are in the correct order?    
# TODO: Should the console output be saved when experiment is finished or stopped?
# TODO: Aster, here is the list of what experiment wants from the optimizer:
#       - At minimum, what info do I have to pass to the optimizer to initialize it?
#       - Default hyperparameter options for default models
#       - Should Experiment take in the optimizer object in order to change the hyperparameters easily?
#       - How to run a single optimization step?
#       - How to access objective functions vals and coords in order to sanity check?
#       - If possible, a log of what optimizer does per optimization step

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
        state: ExperimentalState,
        result_processor: ResultProcessor,
        simulation_runner: SimulationRunner,
        experiment_config: ConfigType,
        batch_manager: Optional[BatchManager] = None
    ) -> None:
        
        self.experiment_config = experiment_config
        self.state = state or ExperimentalState()
        self.result_processor = result_processor
        self.simulation_runner = simulation_runner
        # TODO: Things are becoming messy here;
        #       Need to think about config structure.
        self.batch_manager = BatchManagerFactory.make_batch_manager(
            experiment_mode=self.experiment_config.experiment_mode,
            simulation_runner=self.simulation_runner,
            config=self.batch_manager_config
        ) if batch_manager is None else batch_manager

    def initialize_directory_structure(
            self
    ) -> None:
        ...

    def initialize_configs(
            self
    ) -> None:
        ...

    def initialize_optimizer(
            self
    ) -> None:
        ...

    # TODO: is this redundant?
    def _check_initialization(
            self
    ) -> None:
        
        assert isinstance(self.batch_manager, BatchManager)
        assert isinstance(self.result_processor, ResultProcessor)

    def get_parameters_from_optimizer(
        self
    ) -> List[dict]:
        ...

    def send_objectives_to_optimizer(
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

    def run_optimization_experiment(
            self
    ) -> None:
        
        # initialization
        self.initialize_directory_structure()
        self.initialize_configs()  # how to do this???
        self.initialize_optimizer()
        self._check_initialization()
        
        # run
        for i in self.n_iterations:
            list_of_parameters = self.get_parameters_from_optimizer()
            results = self.batch_manager.run_batch(list_of_parameters)
            objectives = self.result_processor.process(results)
            self._sanity_check(list_of_parameters, results, objectives)
            self.state.update(list_of_parameters, results, objectives)
            self.state.save_to_json(self.path)
            self.send_objectives_to_optimizer(objectives)

    def restart_optimization_experiment(
            self
    ) -> None:
        ...
