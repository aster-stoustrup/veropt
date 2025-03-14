from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypedDict

import torch


class OptimisationMode(Enum):
    initial = 1
    bayesian = 2


class InitialPointsGenerationMode(Enum):
    random = 1


class ObjectiveType(Enum):
    integrated = 1
    interface = 1


# TODO: Write a test to make sure the arguments of this and the dict are the same? (except n_init and n_bayes)
@dataclass
class OptimiserSettings:
    n_initial_points: int
    n_bayesian_points: int
    n_evaluations_per_step: int = 1
    objective_weights: list[float] = None
    normalise: bool = True
    n_points_before_fitting: int = None
    verbose: bool = True
    renormalise_each_step: bool = None
    initial_points_generator: InitialPointsGenerationMode = InitialPointsGenerationMode.random

    def __post_init__(self):
        if  self.n_points_before_fitting is None:
            self.n_points_before_fitting = self.n_initial_points - self.n_evaluations_per_step * 2


class OptimiserSettingsInputDict(TypedDict, total=False):
    n_evaluations_per_step: int
    objective_weights: list[float]
    normalise: bool
    n_points_before_fitting: int
    verbose: bool
    renormalise_each_step: bool


class TensorWithNormalisationFlag:
    def __init__(
            self,
            tensor: torch.Tensor,
            normalised: bool
    ):
        self.tensor = tensor
        self.normalised = deepcopy(normalised)

    def __getitem__(self, item):
        return TensorWithNormalisationFlag(
            tensor=self.tensor[item],
            normalised=self.normalised
        )


@dataclass
class SuggestedPoints:
    coordinates: TensorWithNormalisationFlag
    predicted_values: Optional[TensorWithNormalisationFlag]
    generated_at_step: int
