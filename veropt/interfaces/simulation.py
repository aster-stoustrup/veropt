from pydantic import BaseModel
import json
from typing import Any, List, Union, Optional, Dict
from abc import ABC, abstractmethod
import os

# TODO: Do we stick with TypeDict throughout or can we use BaseModel?
# TODO: Should SimulationResult contain outfile location?
class SimulationResult(BaseModel):
    simulation_id: str
    parameters: Dict[str,float]
    stdout_file: str
    stderr_file: str
    return_code: Optional[int] = None  # Maybe needs to be optional to comply with slurm simulation
    output_file: Optional[str] = None


SimulationResultsDict = Dict[int,Union[SimulationResult,List[SimulationResult]]]


class Simulation(ABC):
    @abstractmethod
    def run(
            self,
            parameters: Dict[str,float]
    ) -> SimulationResult:
        ...


class SimulationRunnerConfig(BaseModel):
    ...


class SimulationRunner(ABC):

    def save_set_up_and_run(
            self,
            simulation_id: str,
            parameters: Dict[str,float],
            run_script_directory: str,
            run_script_filename: str
    ) -> Union[SimulationResult, List[SimulationResult]]:

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
            run_script_filename=run_script_filename
        )

        if isinstance(result, list): 
            for r in result: assert isinstance(r, SimulationResult)
        else: assert isinstance(result, SimulationResult)

        return result

    def _save_parameters(
            self,
            parameters: Dict[str,float],
            parameters_json: str
            ) -> None:

        with open(parameters_json, 'w') as fp:
            json.dump(parameters, fp)

    @abstractmethod
    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: Dict[str,float],
            run_script_directory: str,
            run_script_filename: str
    ) -> Union[SimulationResult, List[SimulationResult]]:
        ...
