from abc import ABC
from typing import Literal, Optional, Self

import botorch
import torch

from veropt.optimiser.objective import CallableObjective


class BotorchPracticeObjective(CallableObjective, ABC):

    def __init__(
            self,
            bounds: list[list[float]],
            n_variables: int,
            n_objectives: int,
            function: botorch.test_functions.SyntheticTestFunction,
            variable_names: Optional[list[str]] = None,
            objective_names: Optional[list[str]] = None
    ):

        variable_names = variable_names or [f"var_{i}" for i in range(1, n_variables + 1)]
        objective_names = objective_names or [f"obj_{i}" for i in range(1, n_objectives + 1)]

        self.function = function

        super().__init__(
            bounds=bounds,
            n_variables=n_variables,
            n_objectives=n_objectives,
            variable_names=variable_names,
            objective_names=objective_names
        )

    def _run(self, parameter_values: torch.Tensor) -> torch.Tensor:

        return self.function(parameter_values)


class Hartmann(BotorchPracticeObjective):

    name = 'hartmann'

    def __init__(
            self,
            n_variables: Literal[3, 4, 6]
    ):

        assert n_variables in [3, 4, 6]

        bounds = [[0.0] * 6, [1.0] * 6]
        n_objectives = 1

        function = botorch.test_functions.Hartmann(negate=True)

        super().__init__(
            bounds=bounds,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=['Hartmann']
        )

    @classmethod
    def from_saved_state(cls, saved_state: dict) -> Self:
        return cls(
            n_variables=saved_state['n_variables']
        )
