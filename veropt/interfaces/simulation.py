from pydantic import BaseModel
from typing import Any, List, Union
import abc

# TODO: Do we stick with TypeDict throughout or can we use BaseModel?
class SimulationResult(BaseModel):
    simulation_id: str
    stdout_file: str
    stderr_file: str
    return_code: int


class Simulation(abc.ABC):
    @abc.abstractmethod
    def run(
            self
    ) -> SimulationResult:
        ...


class SimulationRunnerConfig(BaseModel):
    ...


class SimulationRunner(abc.ABC):

    def save_set_up_and_run(
            self,
            id: str,
            parameters: dict,
            setup_path: str,
            setup_name: str
    ) -> Union[SimulationResult, List[SimulationResult]]:
    
        self._save_parameters

        self.set_up_and_run(
            id=id,
            parameters=parameters,
            setup_path=setup_path,
            setup_name=setup_name
        )

    def _save_parameters(self):
        # TODO: Implement

        raise NotImplementedError

    @abc.abstractmethod
    def set_up_and_run(
            self,
            id: str,
            parameters: dict,
            setup_path: str,
            setup_name: str
    ) -> Union[SimulationResult, List[SimulationResult]]:
        ...
