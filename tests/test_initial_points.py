from veropt.optimiser.initial_points import generate_initial_points_latin_hypercube
import torch
import matplotlib.pyplot as plt


# TODO: implement
def test_generate_initial_points() -> None:
    ...


def test_generate_initial_points_latin_hypercube() -> None:

    bounds = torch.tensor([0., 1.])
    n_initial_points = 10
    n_variables = 2

    initial_points = generate_initial_points_latin_hypercube(
        bounds=bounds,
        n_initial_points=n_initial_points,
        n_variables=n_variables
    )

    assert initial_points.shape == (n_initial_points, n_variables)

    for i in range(n_variables):
        column = initial_points[:, i]
        assert torch.all(column >= bounds[0]) and torch.all(column <= bounds[1])
        assert len(torch.unique(column)) == n_initial_points
