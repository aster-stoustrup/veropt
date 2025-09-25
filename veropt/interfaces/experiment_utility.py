from enum import StrEnum
from typing import Optional, Self
import os

from veropt.interfaces.simulation import SimulationResult, SimulationResultsDict
from veropt.interfaces.utility import Config, create_directory

from pydantic import BaseModel


ParametersDict = dict[int, dict[str, float]]


class Point(BaseModel):
    parameters: dict[str, float]
    state: str
    job_id: Optional[int] = None
    result: Optional[SimulationResult] = None
    objective_values: Optional[dict[str, Optional[float]]] = None


class ExperimentMode(StrEnum):
    local = "local"
    local_slurm = "local_slurm"
    remote_slurm = "remote_slurm"


class ExperimentalState(Config):
    experiment_name: str
    experiment_directory: str
    state_json: str
    points: dict[int, Point]
    next_point: int

    def update(
            self,
            new_point: Point
    ) -> None:

        self.points[self.next_point] = new_point
        self.next_point += 1

    @classmethod
    def make_fresh_state(
            cls,
            experiment_name: str,
            experiment_directory: str,
            state_json: str,
    ) -> Self:

        return cls(
            experiment_name=experiment_name,
            experiment_directory=experiment_directory,
            state_json=state_json,
            points={},
            next_point=0
        )

    def get_results(
            self,
            start_point: int,
            end_point: int
    ) -> SimulationResultsDict:

        points_batch = {point_no: self.points[point_no] for point_no in range(start_point, end_point + 1)}
        results_batch = {point_no: point.result for point_no, point in points_batch.items()}

        for point_no, result in results_batch.items():
            assert result is not None, f"No result found for point {point_no}"

        return results_batch  # type: ignore[return-value] #  Type asserted just above

    def get_parameters(
            self,
            start_point: int,
            end_point: int
    ) -> ParametersDict:

        results = self.get_results(
            start_point=start_point,
            end_point=end_point
        )

        return {point_no: result.parameters for point_no, result in results.items()}

    @property
    def n_points(self) -> int:
        return len(self.points)


class ExperimentConfig(Config):
    experiment_name: str
    parameter_names: list[str]
    parameter_bounds: dict[str, list[float]]
    path_to_experiment: str
    experiment_mode: ExperimentMode
    experiment_directory_name: Optional[str] = None
    run_script_filename: str
    run_script_root_directory: Optional[str] = None
    output_filename: str


class PathManager:
    def __init__(
            self,
            experiment_config: ExperimentConfig
    ):

        self.experiment_config = experiment_config

        self.experiment_directory = self.make_experiment_directory_path()
        self.run_script_root_directory = self.make_run_script_root_directory_path()
        self.results_directory = self.make_results_directory()

        self.experimental_state_json = self.make_experimental_state_json()
        self.suggested_parameters_json = self.make_suggested_parameters_json()
        self.evaluated_objectives_json = self.make_evaluated_objectives_json()

        self.optimiser_state_json = f"{self.experiment_directory}/optimiser_state.json"

    def make_experiment_directory_path(self) -> str:

        if self.experiment_config.experiment_directory_name is not None:
            path = os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_directory_name
            )

        else:
            path = os.path.join(
                self.experiment_config.path_to_experiment,
                self.experiment_config.experiment_name
            )

        create_directory(path)
        return path

    def make_run_script_root_directory_path(self) -> str:

        if self.experiment_config.run_script_root_directory is not None:
            path = self.experiment_config.run_script_root_directory

        else:
            path = os.path.join(
                self.experiment_directory,
                f"{self.experiment_config.experiment_name}_setup"  # better name?
            )

        assert os.path.isdir(path), "Run script root directory not found."
        return path

    @staticmethod
    def make_simulation_id(i: int) -> str:
        return f"point_{i}"

    def make_results_directory(self) -> str:
        path = os.path.join(self.experiment_directory, "results")
        create_directory(path)
        return path

    def make_experimental_state_json(self) -> str:
        return os.path.join(
            self.results_directory,
            f"{self.experiment_config.experiment_name}_experimental_state.json"
        )

    def make_suggested_parameters_json(self) -> str:
        return os.path.join(
            self.results_directory,
            f"{self.experiment_config.experiment_name}_suggested_parameters.json"
        )

    def make_evaluated_objectives_json(self) -> str:
        return os.path.join(
            self.results_directory,
            f"{self.experiment_config.experiment_name}_evaluated_objectives.json"
        )
