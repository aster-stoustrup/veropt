from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Union, Optional, List
import os


class Point(BaseModel):
    parameters: Dict[str, float]
    objective_value: Union[float, List[float]]
    state: Optional[str] = None
    processing_method: Optional[str] = None
    job_id: Optional[int] = None
    output_file: Optional[str] = None


# TODO: Should include timestamps?
class ExperimentalState(BaseModel):
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


class PathManager:
    def __init__(
            self,
            experiment_config: ExperimentConfig
    ) -> None:
        
        self.experiment_config = experiment_config
        self.experiment_directory = self.make_experiment_directory_path()
        self.run_script_directory = self.make_run_script_directory_path()

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
        
    def make_run_script_directory_path(
            self,
    ) -> Path:
        
        if self.experiment_config.run_script_directory is not None:
            return self.experiment_config.run_script_directory
        
        else:
            return os.path.join(
                self.experiment_directory,
                f"{self.experiment_config.experiment_name}_setup"  # better name?
            )
        
    @staticmethod
    def make_result_directory_name(
            index: int,
    ) -> str:
        
        return f"point={index}"
