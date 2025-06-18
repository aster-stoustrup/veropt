import abc
from enum import Enum
from typing import Optional, Union

import torch

from veropt.optimiser.utility import check_incoming_objective_dimensions_fix_1d


class Objective:
    def __init__(
            self,
            bounds: list[list[float]],
            n_variables: int,
            n_objectives: int,
            variable_names: Optional[list[str]] = None,
            objective_names: Optional[list[str]] = None
    ):
        # TODO: Check dimensions of the bounds against n_vars
        self.bounds = torch.tensor(bounds)
        self.n_variables = n_variables
        self.n_objectives = n_objectives

        # TODO: Consider dropping these defaults?
        #   - maybe annoying for callable objective...?
        if variable_names is None:
            self.variable_names = [f"Variable {i}" for i in range(1, n_variables + 1)]
        else:
            self.variable_names = variable_names

        if objective_names is None:
            self.objective_names = [f"Objective {i}" for i in range(1, n_objectives + 1)]
        else:
            self.objective_names = objective_names


class CallableObjective(Objective):
    __metaclass__ = abc.ABCMeta

    def __call__(self, parameter_values: torch.Tensor) -> torch.Tensor:

        objective_values = self._run(
            parameter_values=parameter_values
        )

        objective_values = check_incoming_objective_dimensions_fix_1d(
            objective_values=objective_values,
            n_objectives=self.n_objectives,
            function_name='__call__',
            class_name=self.__class__.__name__
        )

        return objective_values


    @abc.abstractmethod
    def _run(self, parameter_values: torch.Tensor) -> torch.Tensor:
        pass


# TODO: Consider if we want to check that var and obj names match at this level
class InterfaceObjective(Objective):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def save_candidates(
            self,
            suggested_variables: dict[str, torch.Tensor]
    ) -> None:
        pass

    @abc.abstractmethod
    def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pass


class ObjectiveKind(Enum):
    callable = 1
    interface = 2


def determine_objective_type(
        objective: Union[CallableObjective, InterfaceObjective]
) -> ObjectiveKind:

    if issubclass(type(objective), CallableObjective):
        return ObjectiveKind.callable
    elif issubclass(type(objective), InterfaceObjective):
        return ObjectiveKind.interface
    else:
        raise ValueError(
            f"The objective must be a subclass of either {CallableObjective.__name__} or {InterfaceObjective.__name__}."
        )
