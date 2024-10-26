import torch
import abc


class NormaliserType:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, matrix: torch.Tensor):
        pass

    @abc.abstractmethod
    def transform(self, matrix: torch.Tensor):
        pass

    @abc.abstractmethod
    def inverse_transform(self, matrix: torch.Tensor) -> torch.Tensor:
        pass


class NormaliserZeroMeanUnitVariance(NormaliserType):
    def __init__(self, matrix: torch.Tensor, norm_dim=1):
        self.means = matrix.mean(dim=norm_dim)
        self.variances = matrix.var(dim=norm_dim)

    def transform(self, matrix: torch.Tensor):
        return (matrix - self.means[:, None]) / torch.sqrt(self.variances[:, None])

    def inverse_transform(self, matrix):
        return matrix * torch.sqrt(self.variances[:, None]) + self.means[:, None]
