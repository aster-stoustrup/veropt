from abc import ABC, abstractmethod
from typing import List
from veropt.interfaces.simulation import SimulationResult


class ResultProcessor(ABC):
    @abstractmethod
    def process(
            self,
            results: List[SimulationResult]
    ) -> List[float]:
        ...
