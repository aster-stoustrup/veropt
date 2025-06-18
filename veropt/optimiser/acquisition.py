import abc
import functools
from enum import Enum
from typing import Callable, Literal, Optional

import botorch
import numpy as np
import scipy
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from veropt.optimiser.optimiser_utility import get_nadir_point
from veropt.optimiser.utility import (
    check_variable_and_objective_shapes, check_variable_objective_values_matching,
    enforce_amount_of_positional_arguments, unpack_variables_objectives_from_kwargs
)


# TODO: Decide on architecture
#   - How is data shared between acq func and its optimiser
#   - How do we implement dist punish stuff


# TODO: Implement distance punishment optimiser


def _check_input_dimensions[T, **P](
        function: Callable[P, T]
) -> Callable[P, T]:

    @functools.wraps(function)
    def check_dimensions(
            *args: P.args,
            **kwargs: P.kwargs,
    ) -> T:

        enforce_amount_of_positional_arguments(
            received_args=args,
            function=function
        )

        self = args[0]
        assert type(self) is BotorchAcquisitionFunction

        variable_values, objective_values = unpack_variables_objectives_from_kwargs(kwargs)

        if variable_values is None and objective_values is None:
            raise RuntimeError("This decorator was called to check input shapes but found no valid inputs.")

        check_variable_and_objective_shapes(
            n_variables=self.n_variables,
            n_objectives=self.n_objectives,
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


class AcquisitionFunction:

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
    ) -> None:

        self.n_variables = n_variables
        self.n_objectives = n_objectives

        self.function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    @_check_input_dimensions
    def __call__(
            self,
            *,
            variable_values: torch.Tensor
    ) -> torch.Tensor:

        assert self.function is not None, "The acquisition function must receive a model before being called."

        return self.function(variable_values)


class BotorchAcquisitionFunction(AcquisitionFunction):

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def refresh(
            self,
            *,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:

        # This structure is to automatically have the decorator on all implementations of refresh
        self._refresh(
            model=model,
            variable_values=variable_values,
            objective_values=objective_values,
        )

    @abc.abstractmethod
    def _refresh(
            self,
            model: botorch.models.model.Model,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor,
    ) -> None:
        pass


class QLogExpectedHyperVolumeImprovement(BotorchAcquisitionFunction):

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
    ) -> None:

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    def _refresh(
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
            n_variables: int,
            n_objectives: int,
            beta: float = 3.0
    ):

        self.beta = beta

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    def _refresh(
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

    def refresh(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        # Implement this in subclasses if needed, otherwise leave blank

        pass

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


class RefreshSetting(Enum):
    simple = 0
    advanced = 1


class DistancePunishmentSequentialOptimiser(AcquisitionOptimiser):

    def __init__(
            self,
            bounds: torch.Tensor,
            single_step_optimiser: AcquisitionOptimiser,
            alpha: float = 1.0,
            omega: float = 1.0,
            refresh_setting: Literal['simple', 'advanced'] = 'advanced'
    ):

        self.single_step_optimiser = single_step_optimiser

        self.alpha = alpha
        self.omega = omega

        self.scaling: Optional[float] = None

        if refresh_setting == 'simple':
            self.refresh_setting = RefreshSetting.simple
        elif refresh_setting == 'advanced':
            self.refresh_setting = RefreshSetting.advanced
        else:
            raise ValueError(f"'refresh_setting' must be 'simple' or 'advanced', received: {refresh_setting}")

        super().__init__(
            bounds=bounds
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.single_step_optimiser.__class__.__name__})"

    def optimise(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:
        raise NotImplementedError

    def refresh(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        if self.refresh_setting == RefreshSetting.simple:
            self._refresh_scaling_simple(acquisition_function=acquisition_function)

        elif self.refresh_setting == RefreshSetting.advanced:
            self._refresh_scaling_advanced(acquisition_function=acquisition_function)

    def _sample_acq_func(
            self,
            acquisition_function: AcquisitionFunction
    ) -> np.ndarray:
        n_acq_func_samples = 1000
        n_params = self.bounds.shape[1]

        random_coordinates = (
                (self.bounds[1] - self.bounds[0]) * torch.rand(n_acq_func_samples, n_params)
                + self.bounds[0]
        )

        random_coordinates = random_coordinates.unsqueeze(0)

        samples = np.zeros(n_acq_func_samples)

        for coord_ind in range(n_acq_func_samples):
            sample = acquisition_function(
                variable_values=random_coordinates[:, coord_ind:coord_ind+1, :]
            )
            samples[coord_ind] = sample.detach().numpy()  # If this is not detached, it causes a memory leak o:)

        return samples

    def _refresh_scaling_simple(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        acq_func_samples = self._sample_acq_func(acquisition_function=acquisition_function)

        sampled_std = acq_func_samples.std()

        self.scaling = sampled_std

    def _refresh_scaling_advanced(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        acq_func_samples = self._sample_acq_func(acquisition_function=acquisition_function)
        acq_func_samples = np.expand_dims(acq_func_samples, axis=1)

        min_clusters = 1
        min_scored_clusters = 2
        max_clusters = 7

        gaussian_fitters = {
            n_clusters: GaussianMixture(n_components=n_clusters)
            for n_clusters in range(min_clusters, max_clusters + 1)
        }
        scores = {
            n_clusters: 0.0
            for n_clusters in range(min_scored_clusters, max_clusters + 1)
        }

        for n_clusters in range(min_clusters, max_clusters + 1):

            gaussian_fitters[n_clusters].fit(acq_func_samples)

            if n_clusters >= min_scored_clusters:

                predictions = gaussian_fitters[n_clusters].predict(acq_func_samples)

                if np.unique(predictions).size > 1:
                    scores[n_clusters] = silhouette_score(
                        X=acq_func_samples,
                        labels=predictions
                    )
                else:
                    # TODO: Verify that this is okay
                    scores[n_clusters] = 0.0

        # Someone please make a prettier version of this >:)
        best_score_n_clusters = list(scores.keys())[np.array(list(scores.values())).argmax()]
        best_fitter = gaussian_fitters[best_score_n_clusters]

        # TODO: Finetune and test criterion for n_c=1
        if best_fitter.covariances_.max() * 3 > gaussian_fitters[1].covariances_[0]:
            best_score_n_clusters = 1
            best_fitter = gaussian_fitters[best_score_n_clusters]

        top_cluster_ind = best_fitter.means_.argmax()

        self.scaling = 2 * float(np.sqrt(best_fitter.covariances_[top_cluster_ind]))
