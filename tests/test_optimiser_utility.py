import numpy as np
import pytest
import torch

from veropt.optimiser.optimiser_utility import format_input_from_objective, format_output_for_objective, \
    get_best_points, get_pareto_optimal_points


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

    best_point = get_best_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights
    )

    assert best_point is not None

    best_variables, best_values, max_index = (
        best_point['variables'], best_point['objectives'], best_point['index']
    )

    assert best_variables is not None, "Something went wrong in this test. Check set-up."
    assert best_values is not None, "Something went wrong in this test. Check set-up."

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

    best_points = get_best_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights,
        objectives_greater_than=1.0
    )

    assert best_points is not None

    best_variables, best_values, max_index = (
        best_points['variables'], best_points['objectives'], best_points['index']
    )

    assert best_variables is not None, "Something went wrong in this test. Check set-up."
    assert best_values is not None, "Something went wrong in this test. Check set-up."

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

    pareto_optimal_points = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )

    pareto_variables, pareto_values, pareto_indices = (
        pareto_optimal_points['variables'], pareto_optimal_points['objectives'], pareto_optimal_points['indices']
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

    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])

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

    pareto_optimal_points = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights,
        sort_by_max_weighted_sum=True
    )

    pareto_variables, pareto_values, pareto_indices = (
        pareto_optimal_points['variables'], pareto_optimal_points['objectives'], pareto_optimal_points['indices']
    )

    assert torch.equal(true_pareto_variables, pareto_variables)
    assert torch.equal(true_pareto_values, pareto_values)
    assert np.array_equal(true_indices, pareto_indices)


def test_format_input_from_objective() -> None:
    expected_amount_points = 4

    variable_names = ['var_1', 'var_2', 'var_3']
    objective_names = ['obj_1', 'obj_2', 'obj_3']

    new_variable_values = {
        'var_3': torch.tensor([0.2, -0.2, -0.1, 2.2]),
        'var_2': torch.tensor([1.2, -1.4, 1.1, 0.2]),
        'var_1': torch.tensor([0.4, 0.3, 0.7, -0.3]),
    }
    new_objective_values = {
        'obj_2': torch.tensor([0.5, -2.1, 0.3, 1.1]),
        'obj_1': torch.tensor([1.1, 0.2, 2.1, 0.4]),
        'obj_3': torch.tensor([0.2, 0.5, 2.1, 2.2]),
    }

    expected_variable_tensor = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [1.2, -1.4, 1.1, 0.2],
        [0.2, -0.2, -0.1, 2.2],
    ])

    expected_objective_values = torch.tensor([
        [1.1, 0.2, 2.1, 0.4],
        [0.5, -2.1, 0.3, 1.1],
        [0.2, 0.5, 2.1, 2.2]
    ])

    expected_variable_tensor = expected_variable_tensor.T
    expected_objective_values = expected_objective_values.T

    (new_variable_values_tensor, new_objective_values_tensor) = format_input_from_objective(
        new_variable_values=new_variable_values,
        new_objective_values=new_objective_values,
        variable_names=variable_names,
        objective_names=objective_names,
        expected_amount_points=expected_amount_points
    )

    assert torch.equal(expected_variable_tensor, new_variable_values_tensor)
    assert torch.equal(expected_objective_values, new_objective_values_tensor)


def test_format_input_from_objective_too_few_points() -> None:
    expected_amount_points = 4

    variable_names = ['var_1', 'var_2', 'var_3', 'var_4']

    objective_names = ['obj_1', 'obj_2', 'obj_3']

    new_variable_values = {
        'var_3': torch.tensor([0.2, -0.2, -0.1]),
        'var_2': torch.tensor([1.2, -1.4, 1.1]),
        'var_1': torch.tensor([0.4, 0.3, 0.7]),
        'var_4': torch.tensor([-0.5, 0.7, 2.0]),
    }
    new_objective_values = {
        'obj_2': torch.tensor([0.5, -2.1, 0.3]),
        'obj_1': torch.tensor([1.1, 0.2, 2.1]),
        'obj_3': torch.tensor([0.2, 0.5, 2.1]),
    }

    with pytest.raises(AssertionError):
        format_input_from_objective(
            new_variable_values=new_variable_values,
            new_objective_values=new_objective_values,
            variable_names=variable_names,
            objective_names=objective_names,
            expected_amount_points=expected_amount_points
        )


def test_format_output_for_objective() -> None:
    variable_names = ['var_1', 'var_2', 'var_3']

    suggested_variables_tensor = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [1.2, -1.4, 1.1, 0.2],
        [0.2, -0.2, -0.1, 2.2]
    ])

    suggested_variables_tensor = suggested_variables_tensor.T

    expected_suggested_variables_dict = {
        'var_1': torch.tensor([0.4, 0.3, 0.7, -0.3]),
        'var_2': torch.tensor([1.2, -1.4, 1.1, 0.2]),
        'var_3': torch.tensor([0.2, -0.2, -0.1, 2.2]),
    }

    suggested_variables_dict = format_output_for_objective(
        suggested_variables=suggested_variables_tensor,
        variable_names=variable_names,
    )

    # Converting to list makes it sensitive to the order of the keys
    assert list(suggested_variables_dict.keys()) == list(expected_suggested_variables_dict.keys())

    for name, tensor in expected_suggested_variables_dict.items():
        assert torch.equal(suggested_variables_dict[name], tensor)


def test_get_nadir_point() -> None:

    # TODO: Implement

    pass
