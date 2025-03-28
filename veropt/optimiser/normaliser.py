import abc

import torch

from veropt.optimiser.optimiser_utility import DataShape


class Normaliser:
    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
    ):
        pass

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
            matrix: torch.Tensor,
            norm_dim: int = DataShape.index_points
    ):
        self.means = matrix.mean(dim=norm_dim)
        self.variances = matrix.var(dim=norm_dim)

        super().__init__()

    def transform(
            self,
            matrix: torch.Tensor
    ) -> torch.Tensor:

        return (matrix - self.means) / torch.sqrt(self.variances)

    def inverse_transform(
            self,
            matrix: torch.Tensor
    ) -> torch.Tensor:

        return matrix * torch.sqrt(self.variances) + self.means
