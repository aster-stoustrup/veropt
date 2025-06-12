from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Union, Optional, List, Self
import os
from veropt.interfaces.simulation import SimulationResult


class Point(BaseModel):
    parameters: Dict[str, float]
    state: Optional[str] = None
    job_id: Optional[int] = None
    output_file: Optional[str] = None
    result: Optional[Union[SimulationResult,List[SimulationResult]]] = None
    processing_method: Optional[str] = None
    objective_value: Optional[Union[float, List[float]]] = None


class Config(BaseModel):
    # TODO: Better name for this maybe?
    #       Make this class:
    #       - Be able to load itself from json and save itself to json
    #       - Be able to load itself from Self or str
    #       - Raise warnings/errors if paths to source jsons are not found
    #       - Raise warning (?) and make a path if destination json path does not exist
    #       - PS. Check if BaseModel does this already
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
    ) -> Self:
        """Load state from JSON file; returns a fresh state if not found."""
        if not os.path.exists(path):
            return cls()
        
        return cls.parse_file(path)
    

class OptimiserConfig(Config):
    n_evals_per_step: int
    n_iterations: int


# TODO: Should include timestamps?
class ExperimentalState(Config):
    experiment_name: str
    experiment_directory: str
    points: Dict[int, Point] = {}
    next_point: int = 0

    def update(
        self, 
        new_point: Point
    ) -> None:
        
        self.points[self.next_point] = new_point
        self.next_point += 1


class ExperimentConfig(Config):
    experiment_name: str
    parameter_names: List[str]
    parameter_bounds: Dict[str,List[float]]
    path_to_experiment: str
    experiment_mode: str
    experiment_directory_name: Optional[str] = None
    run_script_filename: str
    run_script_root_directory: Optional[str] = None
    output_filename: str


class PathManager:
    def __init__(
            self,
            experiment_config: ExperimentConfig
    ) -> None:
        
        self.experiment_config = experiment_config
        self.experiment_directory = self.make_experiment_directory_path()
        self.run_script_root_directory = self.make_run_script_root_directory_path()
        self.experimental_state_json = self.make_experimental_state_json()

    def make_experiment_directory_path(
            self
    ) -> Path:
    
        if self.experiment_config.experiment_directory_name is not None:
            return os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_directory_name
                )
            
        else:
            return os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_name
                )
        
    def make_run_script_root_directory_path(
            self,
    ) -> Path:
        
        if self.experiment_config.run_script_root_directory is not None:
            return self.experiment_config.run_script_root_directory
        
        else:
            return os.path.join(
                self.experiment_directory,
                f"{self.experiment_config.experiment_name}_setup"  # better name?
            )
        
    @staticmethod
    def make_simulation_id(
            i: int,
    ) -> str:
        
        return f"point={i}"
    
    def make_experimental_state_json(
            self
    ) -> Path:
        
        return os.path.join(
            self.experiment_directory,
            "results",
            f"{self.experiment_config.experiment_name}_experimental_state.json"
        )
