import abc
import functools
from typing import Callable, Self

import torch

from veropt.optimiser.acquisition import BotorchAcquisitionFunction
from veropt.optimiser.acquisition_optimiser import AcquisitionOptimiser
from veropt.optimiser.model import GPyTorchFullModel
from veropt.optimiser.utility import DataShape, PredictionDict, check_variable_and_objective_shapes, \
    check_variable_objective_values_matching, \
    enforce_amount_of_positional_arguments, unpack_variables_objectives_from_kwargs
from veropt.optimiser.saver_loader_utility import SavableClass, rehydrate_object


class Predictor(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    @abc.abstractmethod
    def predict_values(
            self,
            *,
            variable_values: torch.Tensor,
    ) -> PredictionDict:
        pass

    @abc.abstractmethod
    def get_acquisition_values(
            self,
            *,
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
            *,
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

    @abc.abstractmethod
    def check_if_model_is_trained(self) -> bool:
        pass

# TODO: Figure out interface between predictor and optimiser
#   - Note: One awkward thing about this might be when we want to display acq func values?
#       - Do we then check if we're using a botorchPredictor or...?


# TODO: Make sure we refresh acq func before suggesting points
#   - Might want to do some checks to make sure the model is updated on all evaluated points


class BotorchPredictor(Predictor):

    name = 'botorch_predictor'

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

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"model: {str(self.model)},\n"
            f"acquisition function: {str(self.acquisition_function)},\n"
            f"acquisition optimiser: {str(self.acquisition_optimiser)},\n"
            f")"
        )

    @staticmethod
    def _check_input_dimensions[T, **P](
            function: Callable[P, T]
    ) -> Callable[P, T]:

        @functools.wraps(function)
        def check_dimensions(
                *args: P.args,
                **kwargs: P.kwargs,
        ) -> T:

            enforce_amount_of_positional_arguments(
                function=function,
                received_args=args
            )

            assert isinstance(args[0], BotorchPredictor)
            self: BotorchPredictor = args[0]

            variable_values, objective_values = unpack_variables_objectives_from_kwargs(kwargs)

            if variable_values is None and objective_values is None:
                raise RuntimeError("This decorator was called to check input shapes but found no valid inputs.")

            check_variable_and_objective_shapes(
                n_variables=self.model.n_variables,
                n_objectives=self.model.n_objectives,
                function_name=function.__name__,
                class_name=self.__class__.__name__,
                variable_values=variable_values,
                objective_values=objective_values,
            )

            return function(
                *args,
                **kwargs
            )

        return check_dimensions

    @_check_input_dimensions
    def predict_values(
            self,
            *,
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

    def get_acquisition_values(
            self,
            *,
            variable_values: torch.Tensor,
    ) -> torch.Tensor:

        return self.acquisition_function(
            variable_values=variable_values
        )

    def suggest_points(
            self,
            verbose: bool
    ) -> torch.Tensor:

        candidates = self.acquisition_optimiser(self.acquisition_function)

        return candidates

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def update_with_new_data(
            self,
            *,
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

        self.acquisition_optimiser.update_bounds(
            new_bounds=new_bounds
        )

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'model': self.model.gather_dicts_to_save(),
                'acquisition_function': self.acquisition_function.gather_dicts_to_save(),
                'acquisition_optimiser': self.acquisition_optimiser.gather_dicts_to_save()
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        model = rehydrate_object(
            superclass=GPyTorchFullModel,
            name=saved_state['model']['name'],
            saved_state=saved_state['model']['state']
        )

        acquisition_function = rehydrate_object(
            superclass=BotorchAcquisitionFunction,
            name=saved_state['acquisition_function']['name'],
            saved_state=saved_state['acquisition_function']['state']
        )

        acquisition_optimiser = rehydrate_object(
            superclass=AcquisitionOptimiser,
            name=saved_state['acquisition_optimiser']['name'],
            saved_state=saved_state['acquisition_optimiser']['state']
        )

        return cls(
            model=model,
            acquisition_function=acquisition_function,
            acquisition_optimiser=acquisition_optimiser
        )

    def check_if_model_is_trained(self) -> bool:
        return self.model.model_has_been_trained
