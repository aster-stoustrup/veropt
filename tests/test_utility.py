import pytest
import torch

from veropt.utility import NormaliserZeroMeanUnitVariance


def test_standard_normaliser_transform():

    test_matrix = torch.tensor([
        [5, 3, 3.5],
        [8.4, 1.1, 3.2],
        [7.5, 3.4, 2.1]
    ])

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)

    normed_test_matrix = normaliser.transform(test_matrix)

    assert pytest.approx(normed_test_matrix.mean(), abs=1e-6) == 0.0
    assert pytest.approx(normed_test_matrix.var(), abs=1e-6) == 1.0

def test_standard_normaliser_inverse_transform():

    test_matrix = torch.tensor([
        [5, 3, 3.5],
        [8.4, 1.1, 3.2],
        [7.5, 3.4, 2.1]
    ])

    normaliser = NormaliserZeroMeanUnitVariance(matrix=test_matrix)

    normed_test_matrix = normaliser.transform(test_matrix)

    recreated_test_matrix = normaliser.inverse_transform(matrix=normed_test_matrix)

    assert pytest.approx(recreated_test_matrix.mean(), abs=1e-6) == test_matrix.mean()
    assert pytest.approx(recreated_test_matrix.var(), abs=1e-6) == test_matrix.var()


