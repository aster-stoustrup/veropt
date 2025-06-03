import abc
from typing import Callable, Optional

import botorch
import numpy as np
import scipy
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from veropt.optimiser.optimiser_utility import get_nadir_point


# TODO: Decide on architecture
#   - How is data shared between acq func and its optimiser
#   - How do we implement dist punish stuff


# TODO: Implement distance punishment optimiser


class AcquisitionFunction:

    def __init__(self) -> None:
        self.function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def __call__(
            self,
            points: torch.Tensor
    ) -> torch.Tensor:

        assert self.function is not None, "The acquisition function must receive a model before being called."

        return self.function(points)


class BotorchAcquisitionFunction(AcquisitionFunction):

    @abc.abstractmethod
    def refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:
        pass


class QLogExpectedHyperVolumeImprovement(BotorchAcquisitionFunction):

    def __init__(self) -> None:

        super().__init__()

    def refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        nadir_point = get_nadir_point(
            variable_values=variable_values,
            objective_values=objective_values
        )

        partitioning = FastNondominatedPartitioning(
            ref_point=nadir_point,
            Y=objective_values
        )

        self.function = botorch.acquisition.multi_objective.logei.qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=nadir_point,
            partitioning=partitioning
        )


class UpperConfidenceBound(BotorchAcquisitionFunction):

    def __init__(
            self,
            beta: float = 3.0
    ):

        self.beta = beta

        super().__init__()

    def refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        self.function = botorch.acquisition.analytic.UpperConfidenceBound(
            model=model,
            beta=self.beta
        )


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

    def update_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:
        self.bounds = new_bounds

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
            max_iter: int = 1000
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
