import abc
from typing import Self

import torch

from veropt.optimiser.utility import DataShape
from veropt.optimiser.saver_loader_utility import SavableClass


class Normaliser(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    @classmethod
    @abc.abstractmethod
    def from_tensor(
            cls,
            tensor: torch.Tensor,
            norm_dim: int = DataShape.index_points
    ) -> 'NormaliserZeroMeanUnitVariance':

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

    name = 'zero_mean_unit_variance'

    def __init__(
            self,
            means: torch.Tensor,
            variances: torch.Tensor,
    ):
        self.means = means
        self.variances = variances

    @classmethod
    def from_tensor(
            cls,
            tensor: torch.Tensor,
            norm_dim: int = DataShape.index_points
    ) -> 'NormaliserZeroMeanUnitVariance':

        means = tensor.mean(dim=norm_dim)
        variances = tensor.var(dim=norm_dim)

        return cls(
            means=means,
            variances=variances,
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        means = torch.tensor(saved_state['means'])
        variances = torch.tensor(saved_state['variances'])

        return cls(
            means=means,
            variances=variances,
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

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'means': self.means,
                'variances': self.variances,
            }
        }
