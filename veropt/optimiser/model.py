import abc

import torch


class SurrogateModel:
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
            self,
            coordinates: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def train_model(
            self,
            coordinates: torch.Tensor,
            values: torch.Tensor
    ):
        pass
