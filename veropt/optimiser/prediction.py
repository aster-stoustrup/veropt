import abc

import torch

from veropt.optimiser.acquisition import Acquisition, BotorchAcquisitionFunction
from veropt.optimiser.model import GPyTorchFullModel


class Predictor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_values(
            self,
            variable_values: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def suggest_points(
            self,
            verbose: bool
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def update_with_new_data(
            self,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:
        pass

    @abc.abstractmethod
    def update_bounds(
            self,
            new_bounds: torch.Tensor,
    ) -> None:
        pass


# TODO: Figure out interface between predictor and optimiser
#   - Note: One awkward thing about this might be when we want to display acq func values?
#       - Do we then check if we're using a botorchPredictor or...?


# TODO: Make sure we refresh acq func before suggesting points
#   - Might want to do some checks to make sure the model is updated on all evaluated points


class BotorchPredictor(Predictor):
    def __init__(
            self,
            model: GPyTorchFullModel,
            acquisition: BotorchAcquisitionFunction
    ) -> None:
        self.model = model
        self.acquisition = acquisition

        super().__init__()

    def predict_values(
            self,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        # TODO: Implement

        raise NotImplementedError

    def suggest_points(
            self,
            verbose: bool
    ) -> torch.Tensor:

        # TODO: Implement

        raise NotImplementedError

    def update_with_new_data(
            self,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:

        # TODO: Implement

        raise NotImplementedError

    def update_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:

        # TODO: Implement

        raise NotImplementedError


