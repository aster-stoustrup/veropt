import json
from typing import List, Union, Optional, Dict, TypedDict
from abc import ABC, abstractmethod
import os

import torch
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


SimulationResultsDict = Union[dict[int, SimulationResult], dict[int, list[SimulationResult]]]


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

    def save_set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: str,
            output_filename: str
    ) -> Union[SimulationResult, list[SimulationResult]]:

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

        error = TypeError("Simulation must return a SimulationResult or a list of SimulationResults.")

        if isinstance(result, list):
            for r in result:
                assert isinstance(r, SimulationResult), error
        else:
            assert isinstance(result, SimulationResult), error

        return result

    def _save_parameters(
            self,
            parameters: dict[str, float],
            parameters_json: str
    ) -> None:

        with open(parameters_json, 'w') as fp:
            json.dump(parameters, fp)

    @abstractmethod
    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: str,
            output_filename: str
    ) -> Union[SimulationResult, list[SimulationResult]]:
        ...
