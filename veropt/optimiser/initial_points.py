from enum import StrEnum, auto
from typing import Literal

import torch


InitialPointsChoice = Literal['random']


# TODO: Consider if we're keeping this?
class InitialPointsGenerationMode(StrEnum):
    random = auto()


def generate_initial_points_random(
        bounds: torch.Tensor,
        n_initial_points: int,
        n_variables: int
) -> torch.Tensor:

    return (bounds[1] - bounds[0]) * torch.rand(n_initial_points, n_variables) + bounds[0]


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

    else:
        raise ValueError(
            f"Initial point mode {initial_points_generator} is not understood or not implemented."
        )
