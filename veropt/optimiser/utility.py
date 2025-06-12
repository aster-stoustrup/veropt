from typing import Callable, Optional

import torch

from veropt.optimiser.optimiser_utility import DataShape, TensorWithNormalisationFlag


def check_variable_values_shape(
        variable_values: torch.Tensor,
        n_variables: int,
        function_name: str,
        class_name: str
) -> None:
    error_message = (
        f"Tensor 'variable_values' should have shape [n_points, n_variables = {n_variables}] "
        f"but received shape {list(variable_values.shape)} \n"
        f"(in function '{function_name}' in class '{class_name}')."
    )

    if len(variable_values.shape) < 2:
        raise ValueError(error_message)

    if not variable_values.shape[DataShape.index_dimensions] == n_variables:
        raise ValueError(error_message)


def check_objective_values_shape(
        objective_values: torch.Tensor,
        n_objectives: int,
        function_name: str,
        class_name: str
) -> None:

    error_message = (
        f"Tensor 'objective_values' should have shape [n_points, n_objectives = {n_objectives}] "
        f"but received shape {list(objective_values.shape)} \n"
        f"(in function '{function_name}' in class '{class_name}')."
    )

    if len(objective_values.shape) < 2:
        raise ValueError(error_message)

    if not objective_values.shape[DataShape.index_dimensions] == n_objectives:
        raise ValueError(error_message)


def check_variable_and_objective_shapes(
        n_variables: int,
        n_objectives: int,
        function_name: str,
        class_name: str,
        variable_values: Optional[torch.Tensor] = None,
        objective_values: Optional[torch.Tensor] = None
) -> None:

    if variable_values is not None:

        check_variable_values_shape(
            variable_values=variable_values,
            n_variables=n_variables,
            function_name=function_name,
            class_name=class_name,
        )

    if objective_values is not None:

        check_objective_values_shape(
            objective_values=objective_values,
            n_objectives=n_objectives,
            function_name=function_name,
            class_name=class_name,
        )


def check_variable_objective_values_matching[T, **P](
        function: Callable[P, T],
) -> Callable[P, T]:

    def check_shapes(
            *args: P.args,
            **kwargs: P.kwargs
    ) -> T:

        assert 'variable_values' in kwargs, "Tensor 'variable_values' must be specified to use this decorator"
        assert 'objective_values' in kwargs, "Tensor 'objective_values' must be specified to use this decorator"

        variable_values = kwargs['variable_values']
        objective_values = kwargs['objective_values']

        assert type(variable_values) is torch.Tensor, "'variable_values' must be of type torch.Tensor"
        assert type(objective_values) is torch.Tensor, "'objective_values' must be of type torch.Tensor"

        assert len(variable_values.shape) == 2, (
            f"'variable_values' must be of shape [n_points, n_variables] "
            f"but received shape {list(variable_values.shape)} "
        )
        assert len(objective_values.shape) == 2, (
            "'objective_values' must be of shape [n_points, n_objectives] "
            f"but received shape {list(objective_values.shape)} "
        )

        error_message = (
            "The number of points must match between variable values and objective_values. \n"
            f"Got shape [n_points = {variable_values.shape[0]}, n_variables = {variable_values.shape[1]}] "
            f"for variable_values "
            f"and shape [n_points = {objective_values.shape[0]}, n_objectives = {objective_values.shape[1]}] "
            f"for objective_values."
        )

        if not variable_values.shape[DataShape.index_points] == objective_values.shape[DataShape.index_points]:
            raise ValueError(error_message)

        return function(
            *args,
            **kwargs
        )

    return check_shapes


def unpack_variables_objectives_from_kwargs(
        kwargs: dict
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    if 'variable_values' in kwargs:
        variable_values = kwargs['variable_values']
        assert type(variable_values) is torch.Tensor
    else:
        variable_values = None

    if 'objective_values' in kwargs:
        objective_values = kwargs['objective_values']
        assert type(objective_values) is torch.Tensor
    else:
        objective_values = None

    return variable_values, objective_values


def unpack_flagged_variables_objectives_from_kwargs(
        kwargs: dict
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    if 'variable_values_flagged' in kwargs:
        flagged_variable_values = kwargs['variable_values']
        assert type(flagged_variable_values) is TensorWithNormalisationFlag
        variable_values = flagged_variable_values.tensor
    else:
        variable_values = None

    if 'objective_values_flagged' in kwargs:
        flagged_objective_values = kwargs['objective_values']
        assert type(flagged_objective_values) is TensorWithNormalisationFlag
        objective_values = flagged_objective_values.tensor
    else:
        objective_values = None

    return variable_values, objective_values
