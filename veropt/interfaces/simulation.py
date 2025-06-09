from pydantic import BaseModel
import json
from typing import Any, List, Union
import abc
import os

# TODO: Do we stick with TypeDict throughout or can we use BaseModel?
class SimulationResult(BaseModel):
    simulation_id: str
    parameters: dict
    stdout_file: str
    stderr_file: str
    return_code: int


class Simulation(abc.ABC):
    @abc.abstractmethod
    def run(
            self,
            parameters: dict
    ) -> SimulationResult:
        ...


class SimulationRunnerConfig(BaseModel):
    ...


class SimulationRunner(abc.ABC):

    def save_set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict,
            setup_path: str,
            setup_name: str
    ) -> Union[SimulationResult, List[SimulationResult]]:

        parameters_json_filename = f"{simulation_id}_parameters.json"
        parameters_json = os.path.join(setup_path, parameters_json_filename)

        self._save_parameters(
            parameters=parameters,
            parameters_json=parameters_json
        )

        result = self.set_up_and_run(
            simulation_id=simulation_id,
            parameters=parameters,
            setup_path=setup_path,
            setup_name=setup_name
        )

        return result

    def _save_parameters(
            self,
            parameters: dict,
            parameters_json: str
            ) -> None:

        with open(parameters_json, 'w') as fp:
            json.dump(parameters, fp)

    @abc.abstractmethod
    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict,
            setup_path: str,
            setup_name: str
    ) -> Union[SimulationResult, List[SimulationResult]]:
        ...
