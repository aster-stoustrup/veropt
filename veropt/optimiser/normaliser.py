import abc

import torch

from veropt.optimiser.utility import DataShape


class Normaliser:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            tensor: torch.Tensor,
    ) -> None:
        self.tensor = tensor

    @abc.abstractmethod
    def transform(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:

        pass

    @abc.abstractmethod
    def inverse_transform(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:

        pass


class NormaliserZeroMeanUnitVariance(Normaliser):
    def __init__(
            self,
            tensor: torch.Tensor,
            norm_dim: int = DataShape.index_points
    ):
        self.means = tensor.mean(dim=norm_dim)
        self.variances = tensor.var(dim=norm_dim)

        super().__init__(
            tensor=tensor
        )

    def transform(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:

        return (tensor - self.means) / torch.sqrt(self.variances)

    def inverse_transform(
            self,
            tensor: torch.Tensor
    ) -> torch.Tensor:

        return tensor * torch.sqrt(self.variances) + self.means
