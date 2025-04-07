import abc

import torch


class AcquisitionFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self
    ):
        raise NotImplementedError

    abc.abstractmethod
    def suggest_points(self) -> torch.Tensor:
        pass

    def set_bounds(
            self,
            new_bounds: torch.Tensor
    ):
        self.bounds = new_bounds
        raise NotImplementedError("Aster, confirm that nothing else needs to happen here.")




