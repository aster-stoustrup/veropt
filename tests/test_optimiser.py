from typing import Optional, Self

import torch

from veropt.optimiser.constructors import bayesian_optimiser
from veropt.optimiser.objective import InterfaceObjective
from veropt.optimiser.practice_objectives import Hartmann
from veropt.optimiser.utility import DataShape


def test_run_optimisation_step() -> None:
    n_initial_points = 16
    n_bayesian_points = 32

    n_evalations_per_step = 4

    objective = Hartmann(
        n_variables=6
    )

    optimiser = bayesian_optimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evalations_per_step,
        objective=objective,
        verbose=False,
        model={
            'training_settings': {
                'max_iter': 50
            }
        },
        acquisition_optimiser={
            'optimiser': 'dual_annealing',
            'optimiser_settings': {
                'max_iter': 50
            }
        }
    )

    for i in range(5):
        optimiser.run_optimisation_step()

    # TODO: Mostly wanna see if this runs
    #   - Maybe we can figure out something useful to test, otherwise can just start with something that checks
    #    that it's not failing?
    #   - Idea: make very "obvious" opt problem and check suggested bayes point is within range of expected position?
    #       - Though as an integration test, I guess it'll be harder to control all points...?
    #   - Extra idea: Make tests with controlled seed (if possible) so we can test consistency across PR's


def test__set_up_settings_nans() -> None:

    # TODO: Implement once nan's can be handled in veropt.optimiser

    # class TestObjective(InterfaceObjective):
    #     def __init__(self) -> None:
    #         super().__init__(
    #             bounds_lower=[0, 0, 0],
    #             bounds_upper=[1, 1, 1],
    #             n_variables=3,
    #             n_objectives=2,
    #             variable_names=[
    #                 'var_1',
    #                 'var_2',
    #                 'var_3'
    #             ],
    #             objective_names=[
    #                 'obj_1',
    #                 'obj_2'
    #             ]
    #         )
    #
    #     def save_candidates(self, suggested_variables: dict[str, torch.Tensor]) -> None:
    #         pass
    #
    #     def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    #
    #         var_1_tensor = torch.rand(n_evaluations_per_step)
    #         var_2_tensor = torch.rand(n_evaluations_per_step)
    #         var_3_tensor = torch.rand(n_evaluations_per_step)
    #
    #         obj_1_tensor = torch.rand(n_evaluations_per_step) * 4
    #         obj_2_tensor = torch.rand(n_evaluations_per_step) * 6
    #
    #         obj_1_tensor[3] = torch.nan
    #
    #         return (
    #             {
    #                 'var_1': var_1_tensor,
    #                 'var_2': var_2_tensor,
    #                 'var_3': var_3_tensor,
    #             },
    #             {
    #                 'obj_1': obj_1_tensor,
    #                 'obj_2': obj_2_tensor,
    #             }
    #         )
    #
    #     @classmethod
    #     def from_saved_state(
    #             cls,
    #             saved_state: dict
    #     ) -> Self:
    #         raise NotImplementedError
    #
    # n_evaluations_per_step = 4
    #
    # objective = TestObjective()
    #
    # optimiser = bayesian_optimiser(
    #     n_initial_points=4,
    #     n_bayesian_points=8,
    #     n_evaluations_per_step=n_evaluations_per_step,
    #     objective=objective,
    # )
    #
    # optimiser.run_optimisation_step()
    # optimiser.run_optimisation_step()

    pass


def test_add_new_points_interface_objective() -> None:

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

    var_1_tensor = torch.tensor([0.4, 0.1, 0.7, 0.3])
    var_2_tensor = torch.tensor([0.1, 0.5, 0.2, 0.6])
    var_3_tensor = torch.tensor([0.5, 0.2, 0.5, 0.9])

    obj_1_tensor = torch.tensor([2.1, 3.4, 5.2, 1.2])
    obj_2_tensor = torch.tensor([4.3, 6.3, 1.2, 4.2])

    objective = TestObjective()

    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
        model={
            'training_settings': {
                'max_iter': 50
            }
        }
    )

    optimiser.run_optimisation_step()

    expected_var_tensor = torch.stack(
        tensors=(
            var_1_tensor,
            var_2_tensor,
            var_3_tensor,
        ),
        dim=DataShape.index_dimensions
    )

    expected_obj_tensor = torch.stack(
        tensors=(
            obj_1_tensor,
            obj_2_tensor,
        ),
        dim=DataShape.index_dimensions
    )

    assert torch.equal(optimiser._evaluated_variables_real_units, expected_var_tensor)
    assert torch.equal(optimiser._evaluated_objectives_real_units, expected_obj_tensor)


def test_add_new_points_empty_tensors() -> None:
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

            return (
                {
                    'var_1': torch.tensor([]),
                    'var_2': torch.tensor([]),
                    'var_3': torch.tensor([])
                },
                {
                    'obj_1': torch.tensor([]),
                    'obj_2': torch.tensor([]),
                }
            )

        @classmethod
        def from_saved_state(
                cls,
                saved_state: dict
        ) -> Self:
            raise NotImplementedError

    objective = TestObjective()

    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
        model={
            'training_settings': {
                'max_iter': 50
            }
        }
    )

    optimiser.run_optimisation_step()

    assert len(optimiser.evaluated_variable_values) == 0
    assert len(optimiser.evaluated_objective_values) == 0


def test_save_points_interface_objectives() -> None:

    torch.manual_seed(1)

    class TestObjective(InterfaceObjective):
        def __init__(self) -> None:

            self.suggested_variables: Optional[dict] = None

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
            self.suggested_variables = suggested_variables

        def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            return (
                {
                    'var_1': torch.tensor([]),
                    'var_2': torch.tensor([]),
                    'var_3': torch.tensor([])
                },
                {
                    'obj_1': torch.tensor([]),
                    'obj_2': torch.tensor([]),
                }
            )

        @classmethod
        def from_saved_state(
                cls,
                saved_state: dict
        ) -> Self:
            raise NotImplementedError

    objective = TestObjective()

    optimiser = bayesian_optimiser(
        n_initial_points=4,
        n_bayesian_points=8,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
        model={
            'training_settings': {
                'max_iter': 50
            }
        }
    )

    optimiser.run_optimisation_step()

    assert objective.suggested_variables is not None

    for variable_no, variable in enumerate(objective.variable_names):
        assert torch.equal(
            objective.suggested_variables[variable],
            optimiser.suggested_points_history[0].variable_values[:, variable_no]
        )
