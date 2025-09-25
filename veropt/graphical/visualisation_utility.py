from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from plotly.express import colors

from veropt.optimiser.utility import DataShape, PredictionDict


def get_continuous_colour(
        colour_scale: list[list],
        value: float
) -> str:

    if value == 0.0:
        return colour_scale[0][1]
    if value == 1.0:
        return colour_scale[-1][1]

    low_cutoff = None
    low_colour = None

    high_cutoff = None
    high_colour = None

    for cutoff, color in colour_scale:
        if value > cutoff:
            low_cutoff, low_colour = cutoff, color
        else:
            high_cutoff, high_colour = cutoff, color
            break

    assert low_cutoff is not None
    assert low_colour is not None
    assert high_cutoff is not None
    assert high_colour is not None

    intermediate_colour = colors.find_intermediate_color(
        lowcolor=low_colour, highcolor=high_colour,
        intermed=((value - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb"
    )

    return intermediate_colour


def opacity_for_multidimensional_points(
        variable_index: int,
        n_variables: int,
        variable_values: torch.Tensor,
        evaluated_point: torch.Tensor,
        alpha_min: float = 0.1,
        alpha_max: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor]:

    distances_list = []
    index_wo_var_ind = torch.arange(n_variables) != variable_index
    for point_no in range(variable_values.shape[DataShape.index_points]):
        distances_list.append(np.linalg.norm(
            evaluated_point[0, index_wo_var_ind] - variable_values[point_no, index_wo_var_ind]
        ))

    distances = torch.tensor(distances_list)

    normalised_distances = (
        ((distances - distances.min()) / distances.max()) / ((distances - distances.min()) / distances.max()).max()
    )

    normalised_proximity = 1 - normalised_distances

    alpha_values = (alpha_max - alpha_min) * normalised_proximity + alpha_min

    alpha_values[alpha_values.argmax()] = 1.0

    return alpha_values, normalised_distances


class ModelPrediction:
    def __init__(
            self,
            variable_index: int,
            point: torch.Tensor,
            title: str,
            variable_array: np.ndarray,
            predicted_objective_values: PredictionDict,
            acquisition_values: torch.Tensor,
            samples: Optional[torch.Tensor] = None
    ) -> None:

        self.variable_index = variable_index

        self.point = point

        self.title: str = title
        self.variable_array = variable_array
        self.predicted_values_mean = predicted_objective_values['mean']
        self.predicted_values_lower = predicted_objective_values['lower']
        self.predicted_values_upper = predicted_objective_values['upper']
        self.acquisition_values: np.ndarray = acquisition_values.detach().numpy()
        self.samples: Optional[torch.Tensor] = samples

        self.modified_acquisition_values: Optional[list[torch.Tensor]] = None

    def add_modified_acquisition_values(
            self,
            modified_acquisition_values: list[torch.Tensor]
    ) -> None:
        self.modified_acquisition_values = modified_acquisition_values


class ModelPredictionContainer:
    def __init__(self) -> None:
        self.data: list[ModelPrediction] = []
        self.points: torch.Tensor = torch.tensor([])
        self.variable_indices: np.ndarray = np.array([])

    def add_data(
            self,
            model_prediction: ModelPrediction
    ) -> None:

        self.data.append(model_prediction)

        if self.points.numel() == 0:
            self.points = model_prediction.point
        else:
            self.points = torch.concat(
                tensors=[self.points, model_prediction.point],
                dim=0
            )

        self.variable_indices = np.append(
            arr=self.variable_indices,
            values=model_prediction.variable_index
        )

    def __getitem__(
            self,
            data_index: int
    ) -> ModelPrediction:

        return self.data[data_index]

    def locate_data(
            self,
            variable_index: int,
            point: torch.Tensor
    ) -> int | None:

        # Can we do without the mix of np and torch here?
        matching_variable_indices = torch.tensor(np.equal(variable_index, self.variable_indices))

        # NB: Not using any tolerance at the moment, might make this a little unreliable
        no_matching_coordinates_per_point = torch.eq(point, self.points).sum(dim=1)

        n_variables = self.points.shape[DataShape.index_dimensions]

        matching_points = no_matching_coordinates_per_point == n_variables

        matching_point_and_var = matching_variable_indices * matching_points

        full_match_index = torch.where(matching_point_and_var)[0]

        if len(full_match_index) == 1:
            return int(full_match_index)

        elif full_match_index.numel() == 0:
            return None

        elif len(full_match_index) > 1:
            raise RuntimeError("Found more than one matching point.")

        else:
            raise RuntimeError("Unexpected error.")

    def __len__(self) -> int:
        return len(self.data)

    def __call__(
            self,
            variable_index: int,
            point: torch.Tensor
    ) -> ModelPrediction:
        data_ind = self.locate_data(
            variable_index=variable_index,
            point=point
        )

        if data_ind is None:
            raise ValueError("Point not found.")

        return self.data[data_ind]

    def __contains__(
            self,
            point: torch.Tensor
    ) -> bool:

        if len(self) == 0:
            return False

        # Just checking if it has it for var_ind = 0, might be sensible to make it a bit more general/stable
        data_index = self.locate_data(
            variable_index=0,
            point=point
        )

        if data_index is None:
            return False

        elif type(data_index) is int:
            return True

        else:
            raise RuntimeError
