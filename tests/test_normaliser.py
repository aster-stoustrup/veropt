import pytest
import torch

from veropt.optimiser.normalisation import NormaliserZeroMeanUnitVariance
from veropt.optimiser.utility import DataShape


def test_standard_normaliser_transform() -> None:

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

    normaliser = NormaliserZeroMeanUnitVariance.from_tensor(tensor=test_matrix)
    normed_test_matrix = normaliser.transform(tensor=test_matrix)

    mean_tensor = normed_test_matrix.mean(dim=DataShape.index_points)
    assert len(mean_tensor) == n_variables

    for variable_index in range(n_variables):
        assert pytest.approx(mean_tensor[variable_index], abs=1e-6) == 0.0

    variance_tensor = normed_test_matrix.var(dim=DataShape.index_points)
    assert len(variance_tensor) == n_variables

    for variable_index in range(n_variables):
        assert pytest.approx(variance_tensor[variable_index], abs=1e-6) == 1.0


def test_standard_normaliser_inverse_transform() -> None:

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    normaliser = NormaliserZeroMeanUnitVariance.from_tensor(tensor=test_matrix)

    normed_test_matrix = normaliser.transform(test_matrix)

    recreated_test_matrix = normaliser.inverse_transform(tensor=normed_test_matrix)

    mean_tensor = recreated_test_matrix.mean(dim=DataShape.index_points)
    variance_tensor = recreated_test_matrix.var(dim=DataShape.index_points)

    assert pytest.approx(mean_tensor[0], abs=1e-6) == torch.mean(torch.tensor(column_1))
    assert pytest.approx(mean_tensor[1], abs=1e-6) == torch.mean(torch.tensor(column_2))
    assert pytest.approx(mean_tensor[2], abs=1e-6) == torch.mean(torch.tensor(column_3))

    assert pytest.approx(variance_tensor[0], abs=1e-6) == torch.var(torch.tensor(column_1))
    assert pytest.approx(variance_tensor[1], abs=1e-6) == torch.var(torch.tensor(column_2))
    assert pytest.approx(variance_tensor[2], abs=1e-6) == torch.var(torch.tensor(column_3))


def test_standard_normaliser_transform_input_output_shapes() -> None:

    column_1 = [5.2, 3.6, 3.5, 4.3, 1.2]
    column_2 = [8.4, 1.1, 3.2, 5.3, 2.1]
    column_3 = [7.5, 3.4, 2.1, 3.2, 3.1]

    test_matrix = torch.tensor([
        column_1,
        column_2,
        column_3
    ])
    test_matrix = test_matrix.T  # Making the columns columns here

    normaliser = NormaliserZeroMeanUnitVariance.from_tensor(tensor=test_matrix)
    normed_test_matrix = normaliser.transform(test_matrix)

    assert normed_test_matrix.shape == test_matrix.shape


def test_normaliser_integration() -> None:

    # TODO: Implement
    #   - See version in 0.6.1 for inspiration

    pass
