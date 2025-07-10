from abc import ABC, abstractmethod
import os
import xarray as xr
from typing import Union
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
            result: SimulationResult
    ) -> dict[str, float]:
        """Method to calculate objective values from simulation output."""
        ...

    @abstractmethod
    def open_output_file(
            self,
            result: SimulationResult
    ) -> None:
        """Method to open the output file to check if it exists and can be opened."""
        ...

    def process(
            self,
            results: SimulationResultsDict
    ) -> ObjectivesDict:

        objectives_dict: ObjectivesDict = {}

        for i, result in results.items():

            if self._return_nan(result):
                objectives_dict[i] = {name: float('nan') for name in self.objective_names}
            else:
                objectives = self.calculate_objectives(result=result)
                assert [isinstance(objectives[name], float) for name in self.objective_names], \
                    "Objective values must be floats."
                objectives_dict[i] = objectives

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
            objective_names: list[str],
            objectives: dict[str, float],
            fixed_objective: bool = False
    ):

        self.objective_names = objective_names
        self.counter = 1.0
        self.objectives = objectives
        self.fixed_objective = fixed_objective

    def open_output_file(
            self,
            result: SimulationResult
    ) -> None:
        
        result_file = f"{result.output_directory}/{result.output_filename}.txt"

        if "error_output" in result.output_filename:
            raise ValueError("Mock error opening output file.")
        else:
            with open(result_file, "r") as f:
                f.read()

    def calculate_objectives(
            self,
            result: SimulationResult
    ) -> dict[str, float]:
        
        if self.fixed_objective:
            objectives = self.objectives
        else:
            objectives = {name: self.counter for name in self.objective_names}
            self.counter += 1
        return objectives


class TestVerosResultProcessor(ResultProcessor):
    def open_output_file(
            self, 
            result: SimulationResult
    ) -> None:
        
        filename = f"{result.output_filename}.overturning.nc"
        dataset = os.path.join(result.output_directory, filename)
        xr.open_dataset(dataset)

    def calculate_objectives(
            self, 
            result: SimulationResult
    ) -> dict[str, float]:

        filename = f"{result.output_filename}.overturning.nc"
        dataset = os.path.join(result.output_directory, filename)
        ds = xr.open_dataset(dataset)
        amoc_strength = abs(ds.sel(zt=-1000, method="nearest").vsf_depth.min().values * 1e-6)

        return {self.objective_names[0]: amoc_strength}
