import abc
from dataclasses import dataclass
from typing import TypedDict

import torch


class SurrogateModel:
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def train_model(
            self,
            variable_values: torch.Tensor,
            values: torch.Tensor
    ):
        pass


class GPyTorchTrainingParametersInputDict(TypedDict, total=False):
    learning_rate: float
    loss_change_to_stop: float
    max_iter: int
    init_max_iter: int


@dataclass
class GPyTorchTrainingParameters:
    learning_rate: float = 0.1
    loss_change_to_stop: float = 1e-6  # TODO: Find optimal value for this?
    max_iter: int = 1000
    init_max_iter: int = 10000


class GPyTorchModel(SurrogateModel):

    def __init__(self):
        raise NotImplementedError

    def __call__(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def train_model(
            self,
            variable_values: torch.Tensor,
            values: torch.Tensor
    ):
        raise NotImplementedError
