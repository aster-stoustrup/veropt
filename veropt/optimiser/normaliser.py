import abc

import torch


class Normaliser:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(
            self,
            tensor: torch.Tensor
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
            norm_dim: int = 1
    ):

        self.means = matrix.mean(dim=norm_dim)
        self.variances = matrix.var(dim=norm_dim)

    def transform(
            self,
            matrix: torch.Tensor
    ) -> torch.Tensor:

        return (matrix - self.means[:, None]) / torch.sqrt(self.variances[:, None])

    def inverse_transform(
            self,
            matrix: torch.Tensor
    ) -> torch.Tensor:

        return matrix * torch.sqrt(self.variances[:, None]) + self.means[:, None]
