import abc
import functools
from typing import Callable, Optional

import botorch
import torch
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

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
            function=function,
            received_args=args
        )

        assert isinstance(args[0], AcquisitionFunction)
        self: AcquisitionFunction = args[0]

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
            multi_objective: bool
    ) -> None:

        self.n_variables = n_variables
        self.n_objectives = n_objectives

        self.multi_objective = multi_objective

        if self.multi_objective:
            assert self.n_objectives > 1, (
                f"This acquisition function ({self.__class__.__name__}) is meant for multi-objective problems but "
                f"received only {self.n_objectives} objective."
            )
        else:
            assert self.n_objectives == 1, (
                f"This acquisition function ({self.__class__.__name__}) is meant for single-objective problems but "
                f"received {self.n_objectives} objectives."
            )

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
            n_objectives=n_objectives,
            multi_objective=True
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
            n_objectives=n_objectives,
            multi_objective=False
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
