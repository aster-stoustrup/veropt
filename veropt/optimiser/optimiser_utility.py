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
    objective_weights: list[float] = None
    normalise: bool = True
    n_points_before_fitting: int = None
    verbose: bool = True
    renormalise_each_step: bool = None  # TODO: Write a preset for this somewhere
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
    variables: TensorWithNormalisationFlag
    predicted_values: Optional[TensorWithNormalisationFlag]
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


# TODO: Consider splitting into separate functions...?
# TODO: Make a test? :))))
def get_best_points(
        variables: torch.Tensor,
        values: torch.Tensor,
        weights: list[float],
        objectives_greater_than: Optional[float | list[float]] = None,
        best_for_objecive_index: Optional[int] = None
) -> (torch.Tensor, torch.Tensor, int):

    n_objs = values.shape[DataShape.index_dimensions]

    assert objectives_greater_than is None or best_for_objecive_index is None, "Specifying both options not supported"

    if objectives_greater_than is None and best_for_objecive_index is None:

        raise NotImplementedError("Come in with a debugger and fix these dims to the new class")
        max_index = (values * weights).sum(dim=1).argmax()

    elif  objectives_greater_than is not None:

        if type(objectives_greater_than) == float:

            large_enough_objective_values = values > objectives_greater_than

        elif type(objectives_greater_than) == list:

            large_enough_objective_values = values > torch.tensor(objectives_greater_than)

        else:
            raise ValueError

        large_enough_objective_rows = large_enough_objective_values.sum(dim=2) == n_objs

        if large_enough_objective_rows.max() == 0:
            # Could alternatively raise exception but might be overkill
            return None, None

        filtered_objective_values = values * large_enough_objective_rows.unsqueeze(2)

        max_index = (filtered_objective_values * weights).sum(dim=1).argmax()

    elif best_for_objecive_index is not None:
        max_index = values[0, :, best_for_objecive_index].argmax()

    else:
        raise ValueError

    best_variables = variables[0, max_index]
    best_values = values[0, max_index]
    max_index = int(max_index)

    return best_variables, best_values, max_index


def get_pareto_optimal_points(
        variables: torch.Tensor,
        values: torch.Tensor,
        weights: list[float],
        sort_by_max_weighted_sum: bool = True
) -> (torch.Tensor, torch.Tensor, list[bool]):

    pareto_optimal_indices = np.ones(values.shape[0], dtype=bool)
    for value_index, value in enumerate(values):
        if pareto_optimal_indices[value_index]:
            raise NotImplementedError("Come in with a debugger and fix these dims to the new class")
            pareto_optimal_indices[pareto_optimal_indices] = np.any(values[pareto_optimal_indices] > value, axis=1)
            pareto_optimal_indices[value_index] = True

    pareto_optimal_indices = pareto_optimal_indices.nonzero()[0]

    if sort_by_max_weighted_sum:
        pareto_optimal_values = values[pareto_optimal_indices]
        weighted_sum_values = pareto_optimal_values @ np.array(weights)
        sorted_index = weighted_sum_values.argsort()
        sorted_index = np.flip(sorted_index)
        pareto_optimal_indices = pareto_optimal_indices[sorted_index]

    pareto_optimal_indices = pareto_optimal_indices.tolist()

    return (
        variables[:, pareto_optimal_indices],
        values[:, pareto_optimal_indices],
        pareto_optimal_indices
    )
