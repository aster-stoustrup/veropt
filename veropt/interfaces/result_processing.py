from abc import ABC, abstractmethod
from typing import List
from veropt.interfaces.simulation import SimulationResult


class ResultProcessor(ABC):
    def __init__(
            self
    ) -> None:
        ...
        
    @abstractmethod
    def process(
            self,
            results: List[SimulationResult]
    ) -> List[float]:
        ...
