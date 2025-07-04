import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Optional, TypedDict, Unpack

import numpy as np
import scipy
import torch
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from veropt.optimiser.acquisition import AcquisitionFunction
from veropt.optimiser.utility import DataShape, SavableClass, SavableDataClass


class AcquisitionOptimiser(SavableClass):
    __metaclass__ = abc.ABCMeta

    name: str
    maximum_evaluations_per_step: int | None

    def __init__(
            self,
            bounds: torch.Tensor,
            n_evaluations_per_step: int
    ) -> None:

        self.bounds = bounds
        self.n_evaluations_per_step = n_evaluations_per_step

        self.settings: Any  # type: ignore[explicit-any]  # Defined in subclass >:(

        # TODO: Do this in all classes that uses this weird set-up...?
        #   - Otherwise, code might just fail in methods that reference this :(
        #   - Write better error msg
        #   - Could also do a superclass that does all this automatically...?
        #       - That might be rad actually!
        #       - Maybe wait until we get further with the loader and haven't found something better
        assert 'settings' in self.__dict__, "Must define settings in subclass"

        assert 'maximum_evaluations_per_step' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'maximum_evaluations_per_step'."
        )

        if self.maximum_evaluations_per_step is not None:
            assert n_evaluations_per_step == self.maximum_evaluations_per_step, (
                f"This optimiser can only find {self.maximum_evaluations_per_step} point(s) at a time "
                f"but received a setting of {n_evaluations_per_step} evaluations per step."
            )

        assert 'name' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

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

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'settings': self.settings.gather_dicts_to_save()
        }


class TorchNumpyWrapper:
    def __init__(
            self,
            function: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.function = function

    def __call__(self, x: np.ndarray) -> np.ndarray:

        # TODO: Move somewhere prettier:
        #   - And make more general etc etc
        if len(x.shape) == 1:
            x = x.reshape(1, len(x))

        output = self.function(torch.tensor(x))

        return output.detach().numpy()


class DualAnnealingSettingsInputDict(TypedDict, total=False):
    max_iter: int


@dataclass
class DualAnnealingSettings(SavableDataClass):
    max_iter: int = 1_000


class DualAnnealingOptimiser(AcquisitionOptimiser):

    name = 'dual_annealing'
    maximum_evaluations_per_step = 1

    def __init__(
            self,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            **settings: Unpack[DualAnnealingSettingsInputDict]
    ):
        self.settings = DualAnnealingSettings(
            **settings
        )

        super().__init__(
            bounds=bounds,
            n_evaluations_per_step=n_evaluations_per_step
        )

    def optimise(
            self,
            acquisition_function: AcquisitionFunction
    ) -> torch.Tensor:

        wrapped_acquisition_function = TorchNumpyWrapper(
            function=acquisition_function  # type: ignore  # mypy insanity because of the '*' in __call__
        )

        optimisation_result = scipy.optimize.dual_annealing(
            func=wrapped_acquisition_function,
            bounds=self.bounds.T,
            maxiter=self.settings.max_iter
        )

        candidates = torch.tensor(optimisation_result.x)

        return candidates


class ProximityPunishSettingsInputDict(TypedDict, total=False):
    alpha: float
    omega: float
    refresh_setting: Literal['simple', 'advanced']


@dataclass
class ProximityPunishSettings(SavableDataClass):
    alpha: float = 0.7
    omega: float = 1.0
    refresh_setting: Literal['simple', 'advanced'] = 'advanced'


class ProximityPunishmentSequentialOptimiser(AcquisitionOptimiser):

    name = 'proximity_punishment'
    maximum_evaluations_per_step = None

    def __init__(
            self,
            bounds: torch.Tensor,
            n_evaluations_per_step: int,
            single_step_optimiser: AcquisitionOptimiser,
            **settings: Unpack[ProximityPunishSettingsInputDict]
    ):

        self.single_step_optimiser = single_step_optimiser

        self.settings = ProximityPunishSettings(
            **settings
        )

        self.scaling: Optional[float] = None

        super().__init__(
            bounds=bounds,
            n_evaluations_per_step=n_evaluations_per_step
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.single_step_optimiser.__class__.__name__})"

    def optimise(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> torch.Tensor:

        def proximity_punishment_wrapper(
                variable_values: torch.Tensor,
                other_points_variables: list[torch.Tensor]
        ) -> torch.Tensor:

            acquistion_value = acquisition_function(
                variable_values=variable_values
            )

            new_acq_func_val = self._add_proximity_punishment(
                point_variable_values=variable_values,
                acquisition_value=acquistion_value,
                other_points_variable_values=other_points_variables
            )

            return new_acq_func_val

        # TODO: Make acquisition function class with the proximity punishment added
        #   - Satisfies type-hint
        #   - Will be super useful for visualising acq func
        #       - Consider visualisation ease when building

        candidates: list[torch.Tensor] = []

        for candidate_no in range(self.n_evaluations_per_step):

            candidates.append(self.single_step_optimiser(
                acquisition_function=lambda x: proximity_punishment_wrapper(x, candidates)
            ))

            # TODO: Add verbosity flag here
            print(f"Found point {candidate_no + 1} of {self.n_evaluations_per_step}.")

        candidates_tensor = torch.stack(candidates, dim=DataShape.index_points)

        return candidates_tensor

    def refresh(
            self,
            acquisition_function: AcquisitionFunction,
    ) -> None:

        if self.settings.refresh_setting == 'simple':
            self._refresh_scaling_simple(acquisition_function=acquisition_function)

        elif self.settings.refresh_setting == 'advanced':
            self._refresh_scaling_advanced(acquisition_function=acquisition_function)

    def update_bounds(
            self,
            new_bounds: torch.Tensor
    ) -> None:

        self.single_step_optimiser.update_bounds(
            new_bounds=new_bounds
        )

        super().update_bounds(
            new_bounds=new_bounds
        )

    def _add_proximity_punishment(
            self,
            point_variable_values: torch.Tensor,
            acquisition_value: torch.Tensor,
            other_points_variable_values: list[torch.Tensor]
    ) -> torch.Tensor:

        assert self.scaling is not None, "Scaling must have been calculated before adding the proximity punishment."

        proximity_punish = torch.zeros(len(acquisition_value))
        scaling = self.settings.omega * self.scaling

        for other_point_variables in other_points_variable_values:

            proximity_punish += scaling * np.exp(
                -(
                        torch.sum((point_variable_values - other_point_variables) ** 2, dim=1)
                        / (self.settings.alpha ** 2)
                )
            )

        return acquisition_value.detach() - proximity_punish

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

        samples = np.zeros(n_acq_func_samples)

        for coord_ind in range(n_acq_func_samples):
            sample = acquisition_function(
                variable_values=random_coordinates[coord_ind:coord_ind+1, :]
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


    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'settings': self.settings.gather_dicts_to_save(),
            'single_step_optimiser': self.single_step_optimiser.gather_dicts_to_save()
        }
