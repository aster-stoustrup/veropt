from abc import ABC, abstractmethod
from typing import Dict, Union, List
from veropt.interfaces.simulation import SimulationResultsDict


ObjectivesDict = Dict[int,Union[float,List[float]]]


class ResultProcessor(ABC):
    @abstractmethod
    def process(
            self,
            results: SimulationResultsDict
    ) -> ObjectivesDict:
        ...


class MockResultProcessor(ResultProcessor):
    def __init__(
            self,
            objectives_dict: ObjectivesDict
    ) -> None:
        self.objectives_dict = objectives_dict

    def process(
            self,
            results: SimulationResultsDict
    ) -> ObjectivesDict:
        
        return self.objectives_dict