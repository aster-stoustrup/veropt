import functools
import inspect
import json
from copy import deepcopy
from importlib import resources
from typing import Any, Callable, Literal, Mapping, Optional, TypedDict, Union

import torch


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


def count_positional_arguments_in_signature(function: Callable) -> int:

    signature = inspect.signature(function)

    positional_only = inspect.Parameter.POSITIONAL_ONLY
    positional_or_keyword = inspect.Parameter.POSITIONAL_OR_KEYWORD

    return len([
        parameter for parameter in signature.parameters.values() if (
            parameter.kind == positional_only or parameter.kind == positional_or_keyword
        )
    ])


def enforce_amount_of_positional_arguments[T, **P](
        function: Callable[P, T],
        received_args: P.args  # type: ignore  # PEP 612 is boring about this, it's fun and I wanna do it
) -> None:

    n_positional_arguments_orginal_function = count_positional_arguments_in_signature(function)

    if not n_positional_arguments_orginal_function == len(received_args):
        raise TypeError(
            f"{function.__name__}() takes {n_positional_arguments_orginal_function} positional argument "
            f"but {len(received_args)} were given"
        )


def get_arguments_of_function(
        function: Callable,
        argument_type: Literal['all', 'required'] = 'all',
        excluded_arguments: Optional[list[str]] = None
) -> list[str]:

    if argument_type == 'all':
        arguments = _get_all_arguments_of_function(
            function=function
        )

    elif argument_type == 'required':
        arguments = _get_required_arguments_of_function(
            function=function
        )

    else:
        raise ValueError("'argument_type' must be 'all' or 'required'")

    if excluded_arguments is not None:
        for excluded_argument in excluded_arguments:
            for argument_no, argument in enumerate(arguments):
                if argument == excluded_argument:
                    del arguments[argument_no]

    return arguments


def _get_required_arguments_of_function(
        function: Callable
) -> list[str]:

    signature = inspect.signature(function)

    required_arguments = []

    for parameter in signature.parameters.values():
        if parameter.name != 'kwargs':
            if parameter.default is inspect.Parameter.empty:
                required_arguments.append(
                    parameter.name
                )

    return required_arguments


def _get_all_arguments_of_function(
        function: Callable
) -> list[str]:

    signature = inspect.signature(function)

    return [parameter.name for parameter in signature.parameters.values()]


def check_variable_objective_values_matching[T, **P](
        function: Callable[P, T],
) -> Callable[P, T]:

    @functools.wraps(function)
    def check_shapes(
            *args: P.args,
            **kwargs: P.kwargs
    ) -> T:

        enforce_amount_of_positional_arguments(
            function=function,
            received_args=args
        )

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
            "The number of points must match between variable_values and objective_values. \n"
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


def check_incoming_objective_dimensions_fix_1d(
        objective_values: torch.Tensor,
        n_objectives: int,
        function_name: str,
        class_name: str,
) -> torch.Tensor:
    if n_objectives == 1:
        if len(objective_values.shape) == 1:
            objective_values = objective_values.unsqueeze(DataShape.index_dimensions)

    check_objective_values_shape(
        objective_values=objective_values,
        n_objectives=n_objectives,
        function_name=function_name,
        class_name=class_name,
    )

    return objective_values


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
        flagged_variable_values = kwargs['variable_values_flagged']
        assert type(flagged_variable_values) is TensorWithNormalisationFlag
        variable_values = flagged_variable_values.tensor
    else:
        variable_values = None

    if 'objective_values_flagged' in kwargs:
        flagged_objective_values = kwargs['objective_values_flagged']
        assert type(flagged_objective_values) is TensorWithNormalisationFlag
        objective_values = flagged_objective_values.tensor
    else:
        objective_values = None

    return variable_values, objective_values


class PredictionDict(TypedDict):
    mean: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class DataShape:
    index_points = 0
    index_dimensions = 1


class TensorWithNormalisationFlag:
    def __init__(
            self,
            tensor: torch.Tensor,
            normalised: bool
    ):
        self.tensor = tensor
        self.normalised = deepcopy(normalised)

    def __repr__(self) -> str:
        return f"TensorWithNormalisationFlag(normalised={self.normalised}, \n{self.tensor}\n)"

    def __getitem__(
            self,
            item: Union[int, slice, tuple[Union[int, slice], ...]]  # type-hint should technically be as in torch.Tensor
    ) -> 'TensorWithNormalisationFlag':

        return TensorWithNormalisationFlag(
            tensor=self.tensor[item],
            normalised=self.normalised
        )

    def __len__(self) -> int:
        return len(self.tensor)

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape


class MetaClassWithMandatoryAttributes:

    # TODO: Implement

    pass


def _validate_typed_dict(
        typed_dict: Mapping[str, Any],
        expected_typed_dict_class: type,
        object_name: str
) -> None:
    expected_keys = list(expected_typed_dict_class.__annotations__.keys())

    for key in typed_dict.keys():
        assert key in expected_keys, (
            f"Option '{key}' not recognised for '{object_name}'. Expected options: {expected_keys}."
        )


def _load_defaults() -> dict:

    with resources.open_text(
            'veropt',
            'optimiser/default_settings.json'
    ) as defaults_file:
        return json.load(defaults_file)
