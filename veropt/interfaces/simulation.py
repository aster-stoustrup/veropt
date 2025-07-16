import json
from typing import Optional
from abc import ABC, abstractmethod
import os

from pydantic import BaseModel


class SimulationResult(BaseModel):
    simulation_id: str
    parameters: dict[str, float]
    output_directory: str
    output_filename: str
    stdout_file: str
    stderr_file: str
    return_code: Optional[int] = None
    slurm_log_file: Optional[str] = None


SimulationResultsDict = dict[int, SimulationResult]


class Simulation(ABC):
    @abstractmethod
    def run(
            self,
            parameters: dict[str, float]
    ) -> SimulationResult:
        ...


class SimulationRunnerConfig:
    ...


class SimulationRunner(ABC):

    @abstractmethod
    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: str,
            output_filename: str
    ) -> SimulationResult:
        ...

    def save_set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: str,
            output_filename: str
    ) -> SimulationResult:

        parameters_json_filename = f"{simulation_id}_parameters.json"
        parameters_json = os.path.join(run_script_directory, parameters_json_filename)

        self._save_parameters(
            parameters=parameters,
            parameters_json=parameters_json
        )

        result = self.set_up_and_run(
            simulation_id=simulation_id,
            parameters=parameters,
            run_script_directory=run_script_directory,
            run_script_filename=run_script_filename,
            output_filename=output_filename
        )

        assert isinstance(result, SimulationResult), "Simulation must return a SimulationResult."

        return result

    def _save_parameters(
            self,
            parameters: dict[str, float],
            parameters_json: str
    ) -> None:

        with open(parameters_json, 'w') as f:
            json.dump(parameters, f)
