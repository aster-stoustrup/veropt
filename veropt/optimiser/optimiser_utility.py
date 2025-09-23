from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Iterator, Optional, Self, TypedDict, Union

import torch

from veropt.optimiser.initial_points import InitialPointsChoice
from veropt.optimiser.initial_points import InitialPointsGenerationMode
from veropt.optimiser.saver_loader_utility import SavableClass, SavableDataClass
from veropt.optimiser.utility import DataShape, PredictionDict, TensorWithNormalisationFlag


class OptimisationMode(StrEnum):
    initial = auto()
    bayesian = auto()


# TODO: Write a test to make sure the arguments of this and the dict are the same? (except n_init, n_bayes, n_objs)
class OptimiserSettings(SavableClass):

    def __init__(
            self,
            n_initial_points: int,
            n_bayesian_points: int,
            n_objectives: int,
            n_evaluations_per_step: int,
            initial_points_generator: InitialPointsChoice = 'random',
            normalise: bool = True,
            verbose: bool = True,
            renormalise_each_step: Optional[bool] = None,
            n_points_before_fitting: Optional[int] = None,
            objective_weights: Optional[list[float]] = None
    ):
        self.n_initial_points = n_initial_points
        self.n_bayesian_points = n_bayesian_points
        self.n_objectives = n_objectives
        self.n_evaluations_per_step = n_evaluations_per_step

        self.initial_points_generator = InitialPointsGenerationMode(initial_points_generator)

        self.normalise = normalise
        self.verbose = verbose

        if renormalise_each_step is None:

            if n_objectives > 1:
                self.renormalise_each_step = True
            else:
                self.renormalise_each_step = False

        else:
            self.renormalise_each_step = renormalise_each_step

        if n_points_before_fitting is None:
            self.n_points_before_fitting = self.n_initial_points - self.n_evaluations_per_step * 2
        else:
            self.n_points_before_fitting = n_points_before_fitting

        if objective_weights is None:
            self.objective_weights = torch.ones(self.n_objectives) / self.n_objectives
        else:
            self.objective_weights = torch.tensor(objective_weights)

    def gather_dicts_to_save(self) -> dict:

        return self.__dict__

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        return cls(
            **saved_state
        )


class OptimiserSettingsInputDict(TypedDict, total=False):
    objective_weights: list[float]
    normalise: bool
    n_points_before_fitting: int
    verbose: bool
    renormalise_each_step: bool
    initial_points_generator: InitialPointsChoice


@dataclass
class SuggestedPoints(SavableDataClass):
    variable_values: torch.Tensor
    predicted_objective_values: Optional[PredictionDict]
    generated_at_step: int
    generated_with_mode: str
    normalised: bool

    def __getitem__(
            self,
            point_no: int
    ) -> 'SuggestedPoints':

        if self.predicted_objective_values is not None:

            predicted_objective_values = {
                prediction_type: tensor[point_no, :]  # type: ignore[index]  # should be well-defined
                for (prediction_type, tensor) in self.predicted_objective_values.items()
            }

        else:
            predicted_objective_values = None

        return SuggestedPoints(
            variable_values=self.variable_values[point_no, :],
            predicted_objective_values=predicted_objective_values,  # type: ignore[arg-type]  # Should be safe
            generated_at_step=self.generated_at_step,
            generated_with_mode=self.generated_with_mode,
            normalised=self.normalised,
        )

    def __len__(self) -> int:
        return self.variable_values.shape[DataShape.index_points]

    def __iter__(self) -> Iterator['SuggestedPoints']:
        for suggested_point_no in range(len(self)):
            yield self[suggested_point_no]

    @property
    def variable_values_flagged(self) -> TensorWithNormalisationFlag:
        return TensorWithNormalisationFlag(
            tensor=self.variable_values,
            normalised=self.normalised
        )

    def copy(self) -> 'SuggestedPoints':

        if self.predicted_objective_values is None:

            predicted_values_cloned: Optional[PredictionDict] = None

        else:
            predicted_values_cloned = {
                'mean': self.predicted_objective_values['mean'].detach().clone(),
                'lower': self.predicted_objective_values['lower'].detach().clone(),
                'upper': self.predicted_objective_values['upper'].detach().clone(),
            }

        return SuggestedPoints(
            variable_values=self.variable_values.detach().clone(),
            predicted_objective_values=predicted_values_cloned,
            generated_at_step=deepcopy(self.generated_at_step),
            generated_with_mode=deepcopy(self.generated_with_mode),
            normalised=deepcopy(self.normalised),
        )


def list_with_floats_to_string(
        unformatted_list: Union[list[float], list[list[float]]]
) -> str:

    if len(unformatted_list) == 0:
        return "[]"

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


class BestPoints(TypedDict):
    variables: torch.Tensor
    objectives: torch.Tensor
    index: int


def get_best_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: torch.Tensor,
        objectives_greater_than: Optional[float | list[float]] = None,
        best_for_objecive_index: Optional[int] = None
) -> BestPoints | None:

    if len(variable_values) == 0:
        return {
            'variables': torch.tensor([]),
            'objectives': torch.tensor([]),
            'index': -1,
        }

    assert objectives_greater_than is None or best_for_objecive_index is None, "Specifying both options not supported"

    if objectives_greater_than is None and best_for_objecive_index is None:

        max_index_tensor = (objective_values * weights).sum(dim=DataShape.index_dimensions).argmax()
        max_index = int(max_index_tensor)

    elif objectives_greater_than is not None:

        max_index_or_none = _get_points_greater_than(
            objective_values=objective_values,
            weights=weights,
            objectives_greater_than=objectives_greater_than
        )

        if max_index_or_none is None:
            return None
        else:
            max_index = max_index_or_none

    elif best_for_objecive_index is not None:
        max_index_tensor = objective_values[:, best_for_objecive_index].argmax()
        max_index = int(max_index_tensor)

    else:
        raise ValueError

    best_variables = variable_values[max_index]
    best_values = objective_values[max_index]

    return {
        'variables': best_variables,
        'objectives': best_values,
        'index': max_index
    }


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


class ParetoOptimalPoints(TypedDict):
    variables: torch.Tensor
    objectives: torch.Tensor
    indices: list[int]


def get_pareto_optimal_points(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        sort_by_max_weighted_sum: bool = False
) -> ParetoOptimalPoints:

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

    return {
        'variables': variable_values[pareto_optimal_indices_tensor],
        'objectives': objective_values[pareto_optimal_indices_tensor],
        'indices': pareto_optimal_indices
    }


def format_input_from_objective(
        new_variable_values: dict[str, torch.Tensor],
        new_objective_values: dict[str, torch.Tensor],
        variable_names: list[str],
        objective_names: list[str],
        expected_amount_points: int
) -> tuple[torch.Tensor, torch.Tensor]:

    for name in variable_names:
        assert len(new_variable_values[name]) in (expected_amount_points, 0)

    for name in objective_names:
        assert len(new_objective_values[name]) in (expected_amount_points, 0)

    new_variable_values_tensor = torch.stack(
        [new_variable_values[name] for name in variable_names],
        dim=DataShape.index_dimensions
    )

    new_objective_values_tensor = torch.stack(
        [new_objective_values[name] for name in objective_names],
        dim=DataShape.index_dimensions
    )

    return (
        new_variable_values_tensor,
        new_objective_values_tensor
    )


def get_nadir_point(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
) -> torch.Tensor:
    pareto_values = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )['objectives']

    return pareto_values.min(dim=DataShape.index_points)[0]


def format_output_for_objective(
    suggested_variables: torch.Tensor,
    variable_names: list[str]
) -> dict[str, torch.Tensor]:

    suggested_variables_dict = {
        name: suggested_variables[:, variable_number] for (variable_number, name) in enumerate(variable_names)
    }

    return suggested_variables_dict
