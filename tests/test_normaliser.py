import pytest
import torch

from veropt.optimiser.normaliser import NormaliserZeroMeanUnitVariance
from veropt.optimiser.optimiser_utility import DataShape


def test_standard_normaliser_transform():

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    n_variables = test_matrix.shape[DataShape.index_dimensions]

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)
    normed_test_matrix = normaliser.transform(matrix=test_matrix)

    mean_tensor = normed_test_matrix.mean(dim=DataShape.index_points)
    assert len(mean_tensor) == n_variables

    for variable_index in range(n_variables):
        assert pytest.approx(mean_tensor[variable_index], abs=1e-6) == 0.0

    variance_tensor = normed_test_matrix.var(dim=DataShape.index_points)
    assert len(variance_tensor) == n_variables

    for variable_index in range(n_variables):
        assert pytest.approx(variance_tensor[variable_index], abs=1e-6) == 1.0


def test_standard_normaliser_inverse_transform():

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)

    normed_test_matrix = normaliser.transform(test_matrix)

    recreated_test_matrix = normaliser.inverse_transform(matrix=normed_test_matrix)

    mean_tensor = recreated_test_matrix.mean(dim=DataShape.index_points)
    variance_tensor = recreated_test_matrix.var(dim=DataShape.index_points)

    assert pytest.approx(mean_tensor[0], abs=1e-6) == torch.mean(torch.tensor(column_1))
    assert pytest.approx(mean_tensor[1], abs=1e-6) == torch.mean(torch.tensor(column_2))
    assert pytest.approx(mean_tensor[2], abs=1e-6) == torch.mean(torch.tensor(column_3))

    assert pytest.approx(variance_tensor[0], abs=1e-6) == torch.var(torch.tensor(column_1))
    assert pytest.approx(variance_tensor[1], abs=1e-6) == torch.var(torch.tensor(column_2))
    assert pytest.approx(variance_tensor[2], abs=1e-6) == torch.var(torch.tensor(column_3))


def test_standard_normaliser_transform_input_output_shapes():

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)
    normed_test_matrix = normaliser.transform(test_matrix)

    assert normed_test_matrix.shape == test_matrix.shape


# TODO: Update to new system
# def test_normaliser_integration():
#
#     n_init_points = 32
#     n_bayes_points = 12
#
#     n_evals_per_step = 4
#
#     obj_func = PredefinedTestFunction("VehicleSafety")
#
#     optimiser = BayesOptimiser(
#         n_init_points=n_init_points,
#         n_bayes_points=n_bayes_points,
#         obj_func=obj_func,
#         n_evals_per_step=n_evals_per_step,
#         points_before_fitting=n_init_points - n_evals_per_step
#     )
#
#     for i in range(n_init_points//n_evals_per_step):
#         optimiser.run_opt_step()
#
#     assert optimiser.data_has_been_normalised
#
#     obj_means = optimiser.obj_func_vals.mean(dim=1).squeeze(0)
#     obj_stds = optimiser.obj_func_vals.std(dim=1).squeeze(0)
#
#     assert len(obj_means) == optimiser.n_objs
#     assert len(obj_stds) == optimiser.n_objs
#
#     for obj_mean in obj_means:
#         assert obj_mean == pytest.approx(0.0)
#
#     for obj_std in obj_stds:
#         assert obj_std == pytest.approx(1.0)
#
#     param_means = optimiser.obj_func_coords.mean(dim=1).squeeze(0)
#     param_stds = optimiser.obj_func_coords.std(dim=1).squeeze(0)
#
#     assert len(param_means) == optimiser.n_params
#     assert len(param_stds) == optimiser.n_params
#
#     for param_mean in param_means:
#         assert param_mean == pytest.approx(0.0)
#
#     for param_std in param_stds:
#         assert param_std == pytest.approx(1.0)
