import pytest
import torch

from veropt.utility import NormaliserZeroMeanUnitVariance
from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import PredefinedTestFunction


def test_standard_normaliser_transform():

    # Testing a three dimensional tensor for now because this is what's in the code
    #   - In the revamped version, we should be getting rid of that

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    test_matrix = test_matrix[None, :, :]

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)
    normed_test_matrix = normaliser.transform(test_matrix)

    # Testing each column is correct. Could in principle do these in one line each but this might be easier to follow.
    assert pytest.approx(normed_test_matrix.mean(dim=1)[0, 0], abs=1e-6) == 0.0
    assert pytest.approx(normed_test_matrix.mean(dim=1)[0, 1], abs=1e-6) == 0.0
    assert pytest.approx(normed_test_matrix.mean(dim=1)[0, 2], abs=1e-6) == 0.0

    assert pytest.approx(normed_test_matrix.var(dim=1)[0, 0], abs=1e-6) == 1.0
    assert pytest.approx(normed_test_matrix.var(dim=1)[0, 1], abs=1e-6) == 1.0
    assert pytest.approx(normed_test_matrix.var(dim=1)[0, 2], abs=1e-6) == 1.0


def test_standard_normaliser_inverse_transform():

    # Testing a three dimensional tensor for now because this is what's in the code
    #   - In the revamped version, we should be getting rid of that

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    test_matrix = test_matrix[None, :, :]

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)

    normed_test_matrix = normaliser.transform(test_matrix)

    recreated_test_matrix = normaliser.inverse_transform(matrix=normed_test_matrix)

    assert pytest.approx(recreated_test_matrix.mean(dim=1)[0, 0], abs=1e-6) == torch.mean(torch.tensor(column_1))
    assert pytest.approx(recreated_test_matrix.mean(dim=1)[0, 1], abs=1e-6) == torch.mean(torch.tensor(column_2))
    assert pytest.approx(recreated_test_matrix.mean(dim=1)[0, 2], abs=1e-6) == torch.mean(torch.tensor(column_3))

    assert pytest.approx(recreated_test_matrix.var(dim=1)[0, 0], abs=1e-6) == torch.var(torch.tensor(column_1))
    assert pytest.approx(recreated_test_matrix.var(dim=1)[0, 1], abs=1e-6) == torch.var(torch.tensor(column_2))
    assert pytest.approx(recreated_test_matrix.var(dim=1)[0, 2], abs=1e-6) == torch.var(torch.tensor(column_3))


def test_standard_normaliser_transform_input_output_shapes():

    # Testing a three dimensional tensor for now because this is what's in the code
    #   - In the revamped version, we should be getting rid of that

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    test_matrix = test_matrix[None, :, :]

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)
    normed_test_matrix = normaliser.transform(test_matrix)

    assert normed_test_matrix.shape == test_matrix.shape


def test_normaliser_integration():

    n_init_points = 32
    n_bayes_points = 12

    n_evals_per_step = 4

    obj_func = PredefinedTestFunction("VehicleSafety")

    optimiser = BayesOptimiser(
        n_init_points=n_init_points,
        n_bayes_points=n_bayes_points,
        obj_func=obj_func,
        n_evals_per_step=n_evals_per_step,
        points_before_fitting=n_init_points - n_evals_per_step
    )

    for i in range(n_init_points//n_evals_per_step):
        optimiser.run_opt_step()

    assert optimiser.data_normalised

    obj_means = optimiser.obj_func_vals.mean(dim=1).squeeze(0)
    obj_stds = optimiser.obj_func_vals.std(dim=1).squeeze(0)

    assert len(obj_means) == optimiser.n_objs
    assert len(obj_stds) == optimiser.n_objs

    for obj_mean in obj_means:
        assert obj_mean == pytest.approx(0.0)

    for obj_std in obj_stds:
        assert obj_std == pytest.approx(1.0)

    param_means = optimiser.obj_func_coords.mean(dim=1).squeeze(0)
    param_stds = optimiser.obj_func_coords.std(dim=1).squeeze(0)

    assert len(param_means) == optimiser.n_params
    assert len(param_stds) == optimiser.n_params

    for param_mean in param_means:
        assert param_mean == pytest.approx(0.0)

    for param_std in param_stds:
        assert param_std == pytest.approx(1.0)
