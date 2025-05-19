import abc

import torch


class AcquisitionFunction:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def suggest_points(self) -> torch.Tensor:
        pass

    def set_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:
        self.bounds = new_bounds
        # TODO: Check this
        raise NotImplementedError("Aster, confirm that nothing else needs to happen here.")
