import torch


def generate_initial_points_random(
        bounds: torch.Tensor,
        n_initial_points: int,
        n_variables: int
) -> torch.Tensor:

    return (bounds[1] - bounds[0]) * torch.rand(n_initial_points, n_variables) + bounds[0]
