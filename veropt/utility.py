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
    def __init__(self, matrix: torch.Tensor):
        self.mean = matrix.mean()
        self.variance = matrix.var()

    def transform(self, matrix: torch.Tensor):
        # TODO: Probably need to do this per row or something
        #   - Basically needs to be per parameter and per objective
        #   - So need to sort out dimensions
        return (matrix - self.mean) / torch.sqrt(self.variance)

    def inverse_transform(self, matrix):
        return matrix * torch.sqrt(self.variance) + self.mean
