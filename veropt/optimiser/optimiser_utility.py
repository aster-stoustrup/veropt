from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypedDict, Union

import numpy as np
import torch


class DataShape:
    index_points = 0
    index_dimensions = 1


class OptimisationMode(Enum):
    initial = 1
    bayesian = 2


class InitialPointsGenerationMode(Enum):
    random = 1


# TODO: Write a test to make sure the arguments of this and the dict are the same? (except n_init and n_bayes)
@dataclass
class OptimiserSettings:
    n_initial_points: int
    n_bayesian_points: int
    n_evaluations_per_step: int = 1
    objective_weights: Optional[list[float]] = None
    normalise: bool = True
    n_points_before_fitting: Optional[int] = None
    verbose: bool = True
    renormalise_each_step: Optional[bool] = None  # TODO: Write a preset for this somewhere
    initial_points_generator: InitialPointsGenerationMode = InitialPointsGenerationMode.random
    mask_nans = True

    def __post_init__(self):
        if self.n_points_before_fitting is None:
            self.n_points_before_fitting = self.n_initial_points - self.n_evaluations_per_step * 2


class OptimiserSettingsInputDict(TypedDict, total=False):
    n_evaluations_per_step: int
    objective_weights: list[float]
    normalise: bool
    n_points_before_fitting: int
    verbose: bool
    renormalise_each_step: bool
    initial_points_generator: InitialPointsGenerationMode
    mask_nans: bool


class TensorWithNormalisationFlag:
    def __init__(
            self,
            tensor: torch.Tensor,
            normalised: bool
    ):
        self.tensor = tensor
        self.normalised = deepcopy(normalised)

    def __getitem__(
            self,
            item
    ) -> 'TensorWithNormalisationFlag':

        return TensorWithNormalisationFlag(
            tensor=self.tensor[item],
            normalised=self.normalised
        )


@dataclass
class SuggestedPoints:
    variable_values: TensorWithNormalisationFlag
    predicted_objective_values: Optional[TensorWithNormalisationFlag]
    generated_at_step: int


def format_list(
        unformatted_list: Union[list, list[list]]
):
    formatted_list = "["
    if isinstance(unformatted_list[0], list):
        for iteration, list_item in enumerate(unformatted_list):
            for number_ind, number in enumerate(list_item):
                if number_ind < len(list_item) - 1:
                    formatted_list += f"{number:.2f}, "
                elif iteration < len(unformatted_list) - 1:
                    formatted_list += f"{number:.2f}], ["
                else:
                    formatted_list += f"{number:.2f}]"
    else:
        for iteration, list_item in enumerate(unformatted_list):
            if iteration < len(unformatted_list) - 1:
                formatted_list += f"{list_item:.2f}, "
            else:
                formatted_list += f"{list_item:.2f}]"

    return formatted_list


def get_best_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: list[float],
        objectives_greater_than: Optional[float | list[float]] = None,
        best_for_objecive_index: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor, int]:

    weights_tensor = torch.tensor(weights)

    assert objectives_greater_than is None or best_for_objecive_index is None, "Specifying both options not supported"

    if objectives_greater_than is None and best_for_objecive_index is None:

        max_index = (objective_values * weights_tensor).sum(dim=DataShape.index_dimensions).argmax()

    elif objectives_greater_than is not None:

        max_index = _get_points_greater_than(
            objective_values=objective_values,
            weights=weights_tensor,
            objectives_greater_than=objectives_greater_than
        )

        if max_index is None:
            return None, None, None

    elif best_for_objecive_index is not None:
        max_index = objective_values[0, :, best_for_objecive_index].argmax()

    else:
        raise ValueError

    best_variables = variable_values[max_index]
    best_values = objective_values[max_index]
    max_index = int(max_index)

    return best_variables, best_values, max_index


def _get_points_greater_than(
        objective_values: torch.Tensor,
        weights: torch.Tensor,
        objectives_greater_than: Optional[float | list[float]] = None
) -> Union[int, None]:

    n_objs = objective_values.shape[DataShape.index_dimensions]

    if type(objectives_greater_than) is float:

        large_enough_objective_values = objective_values > objectives_greater_than

    elif type(objectives_greater_than) is list:

        large_enough_objective_values = objective_values > torch.tensor(objectives_greater_than)

    else:
        raise ValueError

    large_enough_objective_rows = large_enough_objective_values.sum(dim=DataShape.index_dimensions) == n_objs

    if large_enough_objective_rows.max() == 0:
        # Could alternatively raise exception but might be overkill
        return None

    filtered_objective_values = objective_values * large_enough_objective_rows.unsqueeze(dim=DataShape.index_dimensions)

    max_index = int((filtered_objective_values * weights).sum(dim=DataShape.index_dimensions).argmax())

    return max_index


def get_pareto_optimal_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: Optional[list[float]] = None,
        sort_by_max_weighted_sum: bool = False
) -> (torch.Tensor, torch.Tensor, list[int]):

    pareto_optimal_indices = np.ones(objective_values.shape[DataShape.index_points], dtype=bool)
    for value_index, value in enumerate(objective_values):
        if pareto_optimal_indices[value_index]:
            pareto_optimal_indices[pareto_optimal_indices] = torch.any(
                objective_values[pareto_optimal_indices] > value,
                dim=DataShape.index_dimensions
            )
            pareto_optimal_indices[value_index] = True

    pareto_optimal_indices = pareto_optimal_indices.nonzero()[0]

    if sort_by_max_weighted_sum:

        assert weights is not None, "Must be given weights so sort by weighted sum."

        pareto_optimal_values = objective_values[pareto_optimal_indices]
        weighted_sum_values = pareto_optimal_values @ np.array(weights)
        sorted_index = weighted_sum_values.argsort()
        sorted_index = np.flip(sorted_index)
        pareto_optimal_indices = pareto_optimal_indices[sorted_index]

    pareto_optimal_indices = pareto_optimal_indices.tolist()

    return (
        variable_values[pareto_optimal_indices],
        objective_values[pareto_optimal_indices],
        pareto_optimal_indices
    )
