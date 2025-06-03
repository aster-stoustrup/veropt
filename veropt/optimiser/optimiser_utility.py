from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TypedDict, Union

import torch


class DataShape:
    index_points = 0
    index_dimensions = 1


class OptimisationMode(Enum):
    initial = 1
    bayesian = 2


class InitialPointsGenerationMode(Enum):
    random = 1


# TODO: Write a test to make sure the arguments of this and the dict are the same? (except n_init, n_bayes, n_objs)
class OptimiserSettings:

    def __init__(
            self,
            n_initial_points: int,
            n_bayesian_points: int,
            n_objectives: int,
            n_evaluations_per_step: int = 1,
            initial_points_generator: InitialPointsGenerationMode = InitialPointsGenerationMode.random,
            normalise: bool = True,
            verbose: bool = True,
            renormalise_each_step: Optional[bool] = None,  # TODO: Write a preset for this somewhere
            mask_nans: bool = True,
            n_points_before_fitting: Optional[int] = None,
            objective_weights: Optional[list[float]] = None
    ):
        self.n_initial_points = n_initial_points
        self.n_bayesian_points = n_bayesian_points
        self.n_objectives = n_objectives

        self.n_evaluations_per_step = n_evaluations_per_step

        self.initial_points_generator = initial_points_generator

        self.normalise = normalise
        self.verbose = verbose
        self.renormalise_each_step = renormalise_each_step
        self.mask_nans = mask_nans

        if n_points_before_fitting is None:
            self.n_points_before_fitting = self.n_initial_points - self.n_evaluations_per_step * 2
        else:
            self.n_points_before_fitting = n_points_before_fitting

        if objective_weights is None:
            self.objective_weights = torch.ones(self.n_objectives) / self.n_objectives
        else:
            self.objective_weights = torch.tensor(objective_weights)


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


def list_with_floats_to_string(
        unformatted_list: Union[list[float], list[list[float]]]
) -> str:

    if type(unformatted_list[0]) is list:

        formatted_list = _nested_list_of_floats_to_string(unformatted_list)  # type: ignore[arg-type]

    else:

        formatted_list = "["

        for iteration, list_item in enumerate(unformatted_list):
            if iteration < len(unformatted_list) - 1:
                formatted_list += f"{list_item:.2f}, "
            else:
                formatted_list += f"{list_item:.2f}]"

    return formatted_list


def _nested_list_of_floats_to_string(
        unformatted_list: list[list[float]]
) -> str:
    formatted_list = "["

    for iteration, list_item in enumerate(unformatted_list):
        for number_ind, number in enumerate(list_item):
            if number_ind < len(list_item) - 1:
                formatted_list += f"{number:.2f}, "
            elif iteration < len(unformatted_list) - 1:
                formatted_list += f"{number:.2f}], ["
            else:
                formatted_list += f"{number:.2f}]"

    return formatted_list


def get_best_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: torch.Tensor,
        objectives_greater_than: Optional[float | list[float]] = None,
        best_for_objecive_index: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor, int] | tuple[None, None, None]:

    weights_tensor = torch.tensor(weights)

    assert objectives_greater_than is None or best_for_objecive_index is None, "Specifying both options not supported"

    if objectives_greater_than is None and best_for_objecive_index is None:

        max_index_tensor = (objective_values * weights_tensor).sum(dim=DataShape.index_dimensions).argmax()
        max_index = int(max_index_tensor)

    elif objectives_greater_than is not None:

        max_index_or_none = _get_points_greater_than(
            objective_values=objective_values,
            weights=weights_tensor,
            objectives_greater_than=objectives_greater_than
        )

        if max_index_or_none is None:
            return None, None, None
        else:
            max_index = max_index_or_none

    elif best_for_objecive_index is not None:
        max_index_tensor = objective_values[0, :, best_for_objecive_index].argmax()
        max_index = int(max_index_tensor)

    else:
        raise ValueError

    best_variables = variable_values[max_index]
    best_values = objective_values[max_index]

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
        weights: Optional[torch.Tensor] = None,
        sort_by_max_weighted_sum: bool = False
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:

    pareto_optimal_booleans = torch.ones(
        size=(objective_values.shape[DataShape.index_points],),
        dtype=torch.bool
    )
    for value_index, value in enumerate(objective_values):
        if pareto_optimal_booleans[value_index]:
            pareto_optimal_booleans[pareto_optimal_booleans.clone()] = torch.any(
                input=objective_values[pareto_optimal_booleans] > value,
                dim=DataShape.index_dimensions
            )
            pareto_optimal_booleans[value_index] = True

    pareto_optimal_indices_tensor = pareto_optimal_booleans.nonzero().squeeze()

    if sort_by_max_weighted_sum:

        assert weights is not None, "Must be given weights to sort by weighted sum."

        pareto_optimal_values = objective_values[pareto_optimal_indices_tensor]
        weighted_sum_values = pareto_optimal_values @ weights
        sorted_index = weighted_sum_values.argsort()
        sorted_index = torch.flip(sorted_index, dims=(0,))
        pareto_optimal_indices_tensor = pareto_optimal_indices_tensor[sorted_index]

    pareto_optimal_indices = pareto_optimal_indices_tensor.tolist()

    return (
        variable_values[pareto_optimal_indices_tensor],
        objective_values[pareto_optimal_indices_tensor],
        pareto_optimal_indices
    )


def format_input_from_objective(
        new_variable_values: dict[str, torch.Tensor],
        new_objective_values: dict[str, torch.Tensor],
        variable_names: list[str],
        objective_names: list[str],
        expected_amount_points: int
) -> tuple[torch.Tensor, torch.Tensor]:

    for name in variable_names:
        assert len(new_variable_values[name]) == expected_amount_points

    for name in objective_names:
        assert len(new_objective_values[name]) == expected_amount_points

    new_variable_values_tensor = torch.vstack(
        [new_variable_values[name] for name in variable_names],
    )

    new_objective_values_tensor = torch.vstack(
        [new_objective_values[name] for name in objective_names]
    )

    return (
        new_variable_values_tensor,
        new_objective_values_tensor
    )


def get_nadir_point(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
) -> torch.Tensor:
    _, pareto_values, _ = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )

    return pareto_values.min(dim=DataShape.index_points)[0]


def format_output_for_objective(
    suggested_variables: torch.Tensor,
    variable_names: list[str]
) -> dict[str, torch.Tensor]:

    suggested_variables_dict = {
        name: suggested_variables[variable_number] for (variable_number, name) in enumerate(variable_names)
    }

    return suggested_variables_dict
