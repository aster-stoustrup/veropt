from abc import ABC, abstractmethod
import os
from typing import Dict, Union, List
from veropt.interfaces.simulation import SimulationResult, SimulationResultsDict


ObjectivesDict = dict[int, dict[str, float]]


class ResultProcessor(ABC):

    def __init__(
            self,
            objective_names: list[str]
    ):
        self.objective_names = objective_names

    @abstractmethod
    def calculate_objectives(
            self,
            result: Union[SimulationResult, list[SimulationResult]]
    ) -> dict[str, float]:
        ...

    @abstractmethod
    def open_output_file(
            self,
            result: SimulationResult
    ) -> None:
        ...

    def process(
            self,
            results: SimulationResultsDict
    ) -> ObjectivesDict:

        objectives_dict: ObjectivesDict = {}

        for i, result in results.items():

            if isinstance(result, list):
                filtered_results = [self._return_nan(result=r) for r in result]
                if any(filtered_results):
                    objectives_dict[i] = {name: float('nan') for name in self.objective_names}
                else:
                    objectives_dict[i] = self.calculate_objectives(result=result)

            elif isinstance(result, SimulationResult):
                if self._return_nan(result=result):
                    objectives_dict[i] = {name: float('nan') for name in self.objective_names}
                else:
                    objectives_dict[i] = self.calculate_objectives(result=result)

        return objectives_dict

    def _return_nan(
            self,
            result: SimulationResult
    ) -> bool:

        return_nan = False

        if result.return_code is not None and result.return_code != 0:
            return_nan = True
            print(f"Result {result.simulation_id} has a non-zero return code: {result.return_code}")
        else:
            try:
                self.open_output_file(result=result)
            except Exception as e:
                print(f"Error opening output file for result {result.simulation_id}: {e}")
                return_nan = True

        return return_nan


class MockResultProcessor(ResultProcessor):
    def __init__(
            self,
            objective_names: list[str]
    ):
        self.objective_names = objective_names
        self.counter = 1.0

    def open_output_file(
            self,
            result: SimulationResult
    ) -> None:

        if "error_output" in result.output_filename:
            raise ValueError("Mock error opening output file.")
        else:
            pass

    def calculate_objectives(
            self,
            result: Union[SimulationResult, list[SimulationResult]]
    ) -> dict[str, float]:
        objectives = {name: self.counter for name in self.objective_names}
        self.counter += 1
        return objectives
