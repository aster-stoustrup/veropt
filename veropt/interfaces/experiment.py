from pydantic import BaseModel, Field
from typing import Any, Dict, List
import time
import json
import os
from veropt.interfaces.simulation import SimulationResult, SimulationRunner
from veropt.interfaces.batch_manager import BatchManager

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

class ExperimentalState(BaseModel):
    history: List[Dict[str, Any]] = Field(default_factory=list)
    last_update: float = 0.0

    def update(
        self, 
        params: Dict[str, Any], 
        objective: float, 
        metadata: Dict[str, Any] = None
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
        state: ExperimentalState = None
    ) -> None:
        self.state = state or ExperimentalState()

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
        
    def set_up_batch_of_simulations(
            self,
            list_of_parameters: List[dict]
    ) -> BatchManager:
        ...

    def process_results(
            self, 
            results: List[SimulationResult]
    ) -> List[float]:
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
        
        # run
        for i in self.n_iterations:
            list_of_parameters = self.get_parameters_from_optimizer()
            batch = self.set_up_batch_of_simulations(list_of_parameters)
            results = batch.run_batch()
            objectives = self.process_results(results)
            self._sanity_check(list_of_parameters, results, objectives)
            self.state.update(list_of_parameters, results, objectives)
            self.send_objectives_to_optimizer(objectives)

    def restart_optimization_experiment(
            self
    ) -> None:
        ...
