from enum import StrEnum, auto
from typing import Literal

import torch


InitialPointsChoice = Literal['random', 'latin_hypercube']


# TODO: Consider if we're keeping this?
class InitialPointsGenerationMode(StrEnum):
    random = auto()
    latin_hypercube = auto()


def generate_initial_points_random(
        bounds: torch.Tensor,
        n_initial_points: int,
        n_variables: int
) -> torch.Tensor:

    return (bounds[1] - bounds[0]) * torch.rand(n_initial_points, n_variables) + bounds[0]


def generate_initial_points_latin_hypercube(
        bounds: torch.Tensor,
        n_initial_points: int,
        n_variables: int
) -> torch.Tensor:

    samples = torch.zeros(n_initial_points, n_variables)

    for i in range(n_variables):
        bin_indices = torch.randperm(n_initial_points)
        uniform_samples = (bin_indices.float() + torch.rand(n_initial_points)) / n_initial_points
        samples[:, i] = uniform_samples

    return (bounds[1] - bounds[0]) * samples + bounds[0]


def generate_initial_points(
        initial_points_generator: InitialPointsGenerationMode,
        bounds: torch.Tensor,
        n_initial_points: int,
        n_variables: int
) -> torch.Tensor:

    if initial_points_generator == InitialPointsGenerationMode.random:
        return generate_initial_points_random(
            bounds=bounds,
            n_initial_points=n_initial_points,
            n_variables=n_variables
        )

    elif initial_points_generator == InitialPointsGenerationMode.latin_hypercube:
        return generate_initial_points_latin_hypercube(
            bounds=bounds,
            n_initial_points=n_initial_points,
            n_variables=n_variables
        )

    else:
        raise ValueError(
            f"Initial point mode {initial_points_generator} is not understood or not implemented."
        )
