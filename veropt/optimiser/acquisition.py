import abc
from typing import Any, Callable, Optional, TypeVar

import botorch
import numpy as np
import scipy
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from veropt.optimiser.model import GPyTorchFullModel, SurrogateModel
from veropt.optimiser.optimiser_utility import get_nadir_point


# TODO: Decide on architecture
#   - Where does the acq func optimiser live?
#   - How is data shared between acq func and its optimiser
#   - How do we implement dist punish stuff


class AcquisitionFunction:

    def __init__(self) -> None:
        self.function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def __call__(
            self,
            points: torch.Tensor
    ) -> torch.Tensor:

        assert self.function is not None, "The acquisition function must receive data before being called."

        return self.function(points)


class BotorchAcquisitionFunction:

    def __init__(
            self,
            function_class: type[botorch.acquisition.AcquisitionFunction],
            static_parameters: Optional[dict[str, float]] = None
    ):

        self.function_class = function_class

        self.static_parameters = static_parameters or {}

    def refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        dynamic_parameters = self.calculate_dynamic_parameters(
            variable_values=variable_values,
            objective_values=objective_values,
        )

        self.function = self.function_class(
            model=model.get_gpytorch_model(),
            **self.static_parameters,
            **dynamic_parameters
        )

    @abc.abstractmethod
    def calculate_dynamic_parameters(
            self,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> dict[str, Any]:
        pass


class AcquisitionBotorchqLogEHVI(BotorchAcquisitionFunction):

    def __init__(self) -> None:
        super().__init__(
            function_class=botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement
        )

    def calculate_dynamic_parameters(
            self,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> dict[str, Any]:

        dynamic_parameters = {}

        nadir_point = get_nadir_point(
            variable_values=variable_values,
            objective_values=objective_values
        )

        dynamic_parameters['ref_point'] = nadir_point

        dynamic_parameters['partitioning'] = FastNondominatedPartitioning(
            ref_point=nadir_point,
            Y=objective_values
        )

        return dynamic_parameters


class AcquisitionOptimiser:
    def __init__(
            self,
            bounds: torch.Tensor,
    ) -> None:
        self.bounds = bounds

    def __call__(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:
        return self.optimise(acquisition_function)

    @abc.abstractmethod
    def optimise(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:
        pass


class TorchNumpyWrapper:
    def __init__(
            self,
            function: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.function = function

    def __call__(self, x: np.ndarray) -> np.ndarray:

        output = self.function(torch.tensor(x))

        return output.detach().numpy()


class DualAnnealingOptimiser(AcquisitionOptimiser):

    def __init__(
            self,
            bounds: torch.Tensor,
            max_iter: int,
    ):
        self.max_iter = max_iter

        super().__init__(
            bounds=bounds
        )

    def optimise(
            self,
            acquisition_function: AcquisitionFunction
    ) -> torch.Tensor:

        wrapped_acquisition_function = TorchNumpyWrapper(
            function=acquisition_function
        )

        optimisation_result = scipy.optimize.dual_annealing(
            func=wrapped_acquisition_function,
            bounds=self.bounds.T,
            maxiter=self.max_iter
        )

        candidates = torch.tensor(optimisation_result.x)

        return candidates


class Acquisition:

    def __init__(
            self,
            function: AcquisitionFunction,
            optimiser: AcquisitionOptimiser,
    ) -> None:
        self.function = function
        self.optimiser = optimiser

    def suggest_points(self) -> torch.Tensor:
        return self.optimiser(self.function)

    def set_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:
        self.bounds = new_bounds
        # TODO: Check this
        raise NotImplementedError("Aster, confirm that nothing else needs to happen here.")
