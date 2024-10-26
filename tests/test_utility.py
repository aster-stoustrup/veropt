import pytest
import torch

from veropt.utility import NormaliserZeroMeanUnitVariance


def test_standard_normaliser_transform():

    # Testing a three dimensional tensor here because that's what happens in the optimiser for some reason
    #   - Might be botorch who wanted/wants this

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

    # Testing a three dimensional tensor here because that's what happens in the optimiser for some reason
    #   - Might be botorch who wanted/wants this

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

    # Testing a three dimensional tensor here because that's what happens in the optimiser for some reason
    #   - Might be botorch who wanted/wants this
    #   - Actually probably because the first dimension is the batch size for botorch or something like that
    #   - Could consider if this should be changed in veropt (and handled with a wrapper or something)

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
