import abc
from enum import Enum
from typing import Union

import torch

from veropt.optimiser.saver_loader_utility import SavableClass
from veropt.optimiser.utility import check_incoming_objective_dimensions_fix_1d


class Objective(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self,
            bounds_lower: list[float],
            bounds_upper: list[float],
            n_variables: int,
            n_objectives: int,
            variable_names: list[str],
            objective_names: list[str]
    ):
        assert len(bounds_lower) == n_variables
        assert len(bounds_upper) == n_variables

        self.bounds = torch.tensor([bounds_lower, bounds_upper])
        self.n_variables = n_variables
        self.n_objectives = n_objectives

        self.variable_names = variable_names
        self.objective_names = objective_names

    def gather_dicts_to_save(self) -> dict:
        return {
            'name': self.name,
            'state': {
                'bounds': self.bounds,
                'n_variables': self.n_variables,
                'n_objectives': self.n_objectives,
                'variable_names': self.variable_names,
                'objective_names': self.objective_names,
            }
        }


class CallableObjective(Objective, metaclass=abc.ABCMeta):

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
class InterfaceObjective(Objective, metaclass=abc.ABCMeta):

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

    if isinstance(objective, CallableObjective):
        return ObjectiveKind.callable
    elif isinstance(objective, InterfaceObjective):
        return ObjectiveKind.interface
    else:
        raise ValueError(
            f"The objective must be a subclass of either {CallableObjective.__name__} or {InterfaceObjective.__name__}."
        )
