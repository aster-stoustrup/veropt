import abc
from enum import Enum
from typing import Union

import torch


class Objective:
    def __init__(
            self,
            bounds: torch.Tensor,
            n_variables: int,
            n_objectives: int,
            variable_names: list[str] = None,
            objective_names: list[str] = None
    ):
        self.bounds = bounds
        self.n_variables = n_variables
        self.n_objectives = n_objectives

        if variable_names is None:
            self.variable_names = [f"Variable {i}" for i in range(1, n_variables + 1)]
        else:
            self.variable_names = variable_names

        if objective_names is None:
            self.objective_names = [f"Objective {i}" for i in range(1, n_objectives + 1)]
        else:
            self.objective_names = objective_names


class IntegratedObjective(Objective):

    @abc.abstractmethod
    def run(self, parameter_values):
        pass


class InterfaceObjective(Objective):

    @abc.abstractmethod
    def save_candidates(self):
        pass

    @abc.abstractmethod
    def load_evaluated_points(self):
        pass


class ObjectiveKind(Enum):
    integrated = 1
    interface = 2


def determine_objective_type(
        objective: Union[IntegratedObjective, InterfaceObjective]
) -> ObjectiveKind:

    if issubclass(type(objective), IntegratedObjective):
        return ObjectiveKind.integrated
    elif issubclass(type(objective), InterfaceObjective):
        return ObjectiveKind.interface
    else:
        raise ValueError("The objective must be a subclass of either IntegratedObjective or InterfaceObjective.")
