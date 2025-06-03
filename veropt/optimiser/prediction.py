import abc
from typing import TypedDict

import torch

from veropt.optimiser.acquisition import AcquisitionOptimiser, BotorchAcquisitionFunction
from veropt.optimiser.model import GPyTorchFullModel
from veropt.optimiser.optimiser_utility import DataShape


# TODO: If PEP 764 is accepted, convert this to inline
class PredictionDict(TypedDict):
    mean: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class Predictor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_values(
            self,
            variable_values: torch.Tensor,
    ) -> PredictionDict:
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
            acquisition_function: BotorchAcquisitionFunction,
            acquisition_optimiser: AcquisitionOptimiser
    ) -> None:

        self.model = model
        self.acquisition_function = acquisition_function
        self.acquisition_optimiser = acquisition_optimiser

        super().__init__()

    def predict_values(
            self,
            variable_values: torch.Tensor
    ) -> PredictionDict:

        model_output = self.model(
            variable_values=variable_values,
        )

        n_points = variable_values.shape[DataShape.index_points]

        model_mean = torch.zeros(size=[n_points, self.model.n_objectives])
        model_lower = torch.zeros(size=[n_points, self.model.n_objectives])
        model_upper = torch.zeros(size=[n_points, self.model.n_objectives])

        for objective_no in range(self.model.n_objectives):
            model_mean[:, objective_no] = model_output[objective_no].loc
            model_lower[:, objective_no], model_upper[:, objective_no] = (
                model_output[objective_no].confidence_region()
            )

        return {
            'mean': model_mean,
            'lower': model_lower,
            'upper': model_upper
        }

    def suggest_points(
            self,
            verbose: bool
    ) -> torch.Tensor:

        candidates = self.acquisition_optimiser(self.acquisition_function)

        return candidates

    def update_with_new_data(
            self,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:

        self.model.train_model(
            variable_values=variable_values,
            objective_values=objective_values
        )

        self.acquisition_function.refresh(
            model=self.model.get_gpytorch_model(),
            variable_values=variable_values,
            objective_values=objective_values
        )

    def update_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:

        self.acquisition_optimiser.update_bounds(
            new_bounds=new_bounds
        )

        # TODO: Check if anything else need to happen here

        raise NotImplementedError


