import numpy as np
import torch

from veropt.optimiser.optimiser_utility import get_best_points, get_pareto_optimal_points


def test_get_best_points_simple() -> None:

    variable_values = torch.tensor([
        [0.4, 0.3, 0.7],
        [0.4, 2.4, 0.2],
        [0.1, 1.2, -0.4],
        [3.5, 0.6, 2.1]
    ])
    objective_values = torch.tensor([
        [1.2, 0.5],
        [2.3, 3.4],
        [0.3, 0.5],
        [1.2, 1.4]
    ])

    weights = torch.tensor([0.5, 0.5])

    true_max_index = 1

    best_variables, best_values, max_index = get_best_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights
    )

    assert best_variables is not None, "Something went wrong in this test. Check set-up."
    assert best_values is not None, "Something went wrong in this test. Check set-up."

    # Internally converting to tensor but shouldn't convert input
    assert type(weights) is list

    assert max_index == true_max_index
    assert torch.equal(best_variables, torch.tensor([0.4, 2.4, 0.2]))
    assert torch.equal(best_values, torch.tensor([2.3, 3.4]))


def test_get_best_points_w_objectives_greater_than() -> None:

    variable_values = torch.tensor([
        [0.4, 0.3, 0.7],
        [0.4, 2.4, 0.2],
        [0.1, 1.2, -0.4],
        [3.5, 0.6, 2.1]
    ])
    objective_values = torch.tensor([
        [1.2, 0.5],
        [0.8, 3.4],
        [0.3, 0.5],
        [1.2, 1.4]
    ])

    weights = torch.tensor([0.5, 0.5])

    true_max_index = 3  # Because we're requiring obj>1

    best_variables, best_values, max_index = get_best_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights,
        objectives_greater_than=1.0
    )

    assert best_variables is not None, "Something went wrong in this test. Check set-up."
    assert best_values is not None, "Something went wrong in this test. Check set-up."

    # Internally converting to tensor but shouldn't convert input
    assert type(weights) is list

    assert max_index == true_max_index
    assert torch.equal(best_variables, torch.tensor([3.5, 0.6, 2.1]))
    assert torch.equal(best_values, torch.tensor([1.2, 1.4]))


def test_get_pareto_optimal_points() -> None:

    variable_values = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [0.4, 2.4, 0.2, 0.3],
        [0.1, 1.2, -0.4, 0.5],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])
    objective_values = torch.tensor([
        [1.1, 0.5, 0.3],
        [2.3, 1.2, 0.7],
        [0.3, 0.5, 0.6],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    true_pareto_variables = torch.tensor([
        [0.4, 2.4, 0.2, 0.3],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])

    true_pareto_values = torch.tensor([
        [2.3, 1.2, 0.7],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    true_indices = [1, 3, 4]

    pareto_variables, pareto_values, pareto_indices = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )

    assert torch.equal(true_pareto_variables, pareto_variables)
    assert torch.equal(true_pareto_values, pareto_values)
    assert np.array_equal(true_indices, pareto_indices)


def test_get_pareto_optimal_points_weights() -> None:

    variable_values = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [0.4, 2.4, 0.2, 0.3],
        [0.1, 1.2, -0.4, 0.5],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])
    objective_values = torch.tensor([
        [1.1, 0.5, 0.3],
        [2.3, 1.2, 0.7],
        [0.3, 0.5, 0.6],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    weights = torch.tensor([1/3, 1/3, 1/3])

    true_pareto_variables = torch.tensor([
        [0.4, 2.4, 0.2, 0.3],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])

    true_pareto_values = torch.tensor([
        [2.3, 1.2, 0.7],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    true_indices = [1, 3, 4]

    pareto_variables, pareto_values, pareto_indices = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights,
        sort_by_max_weighted_sum=True
    )

    assert torch.equal(true_pareto_variables, pareto_variables)
    assert torch.equal(true_pareto_values, pareto_values)
    assert np.array_equal(true_indices, pareto_indices)
