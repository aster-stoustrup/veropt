from typing import Self

import torch

from veropt.optimiser.constructors import bayesian_optimiser
from veropt.optimiser.objective import InterfaceObjective


class TestObjective(InterfaceObjective):
    def __init__(self) -> None:
        super().__init__(
            bounds_lower=[0, 0, 0],
            bounds_upper=[1, 1, 1],
            n_variables=3,
            n_objectives=2,
            variable_names=[
                'var_1',
                'var_2',
                'var_3'
            ],
            objective_names=[
                'obj_1',
                'obj_2'
            ]
        )

    def save_candidates(self, suggested_variables: dict[str, torch.Tensor]) -> None:
        pass

    def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        var_1_tensor = torch.rand(n_evaluations_per_step)
        var_2_tensor = torch.rand(n_evaluations_per_step)
        var_3_tensor = torch.rand(n_evaluations_per_step)

        obj_1_tensor = torch.rand(n_evaluations_per_step) * 4
        obj_2_tensor = torch.rand(n_evaluations_per_step) * 6

        obj_1_tensor[3] = torch.nan

        return (
            {
                'var_1': var_1_tensor,
                'var_2': var_2_tensor,
                'var_3': var_3_tensor,
            },
            {
                'obj_1': obj_1_tensor,
                'obj_2': obj_2_tensor,
            }
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        raise NotImplementedError


n_evaluations_per_step = 4

objective = TestObjective()

optimiser = bayesian_optimiser(
    n_initial_points=4,
    n_bayesian_points=8,
    n_evaluations_per_step=n_evaluations_per_step,
    objective=objective,
    model={
        'training_settings': {
            'max_iter': 500  # This is just to develop faster, probably not enough to train well
        }
    },
    acquisition_optimiser={
        'optimiser': 'dual_annealing',
        'optimiser_settings': {
            'max_iter': 300
        }
    }
)

optimiser.run_optimisation_step()

pass

# TODO: Implement
#   - Not sure about current implementation of handling nans
#   - Might need to use as a context manager during training?
