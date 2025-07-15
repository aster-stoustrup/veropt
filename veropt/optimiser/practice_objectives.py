import abc
from typing import Literal, Optional, Self

import botorch
import torch

from veropt.optimiser.objective import CallableObjective


class BotorchPracticeObjective(CallableObjective, metaclass=abc.ABCMeta):

    def __init__(
            self,
            bounds_lower: list[float],
            bounds_upper: list[float],
            n_variables: int,
            n_objectives: int,
            function: botorch.test_functions.base.BaseTestProblem,
            variable_names: Optional[list[str]] = None,
            objective_names: Optional[list[str]] = None
    ):

        variable_names = variable_names or [f"var_{i}" for i in range(1, n_variables + 1)]
        objective_names = objective_names or [f"obj_{i}" for i in range(1, n_objectives + 1)]

        self.function = function

        super().__init__(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
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

        n_objectives = 1

        function = botorch.test_functions.Hartmann(negate=True)

        super().__init__(
            bounds_lower=[0.0] * n_variables,
            bounds_upper=[1.0] * n_variables,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=['Hartmann']
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
            n_variables=saved_state['n_variables']
        )


class VehicleSafety(BotorchPracticeObjective):

    name = 'vehicle_safety'

    def __init__(
            self
    ) -> None:
        n_variables = 5
        n_objectives = 3
        function = botorch.test_functions.VehicleSafety()
        objective_names = [f"VeSa {obj_no + 1}" for obj_no in range(n_objectives)]

        super().__init__(
            bounds_lower=[1.0] * n_variables,
            bounds_upper=[3.0] * n_variables,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=objective_names
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
        )


class DTLZ1(BotorchPracticeObjective):

    name = 'dtlz_1'

    def __init__(
            self,
            n_variables: int = 10,
            n_objectives: int = 5
    ):

        function = botorch.test_functions.DTLZ1(
            dim=n_variables,
            num_objectives=n_objectives,
            negate=True
        )

        objective_names = [f"DTLZ1 {obj_no + 1}" for obj_no in range(n_objectives)]

        super().__init__(
            bounds_lower=[0.0] * n_variables,
            bounds_upper=[1.0] * n_variables,
            n_variables=n_variables,
            n_objectives=n_objectives,
            function=function,
            objective_names=objective_names
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        return cls(
            n_variables=saved_state['n_variables'],
            n_objectives=saved_state['n_objectives']
        )
