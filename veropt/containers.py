from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SuggestedPoint:
    coordinates: torch.tensor
    predicted_values: torch.tensor
    predicted_values_upper: torch.tensor
    predicted_values_lower: torch.tensor


class ModelPrediction:
    def __init__(self, calc_pred_output: tuple, var_ind: int):

        self.var_ind = var_ind

        # TODO: Should do this nicer
        #   - If calc_pred should actually output all this, let's make it a dict?
        self.title: str = calc_pred_output[0]
        self.var_arr: np.ndarray =  calc_pred_output[1]
        self.model_mean_list: list[torch.tensor] = calc_pred_output[2]
        self.model_lower_std_list: list[torch.tensor] = calc_pred_output[3]
        self.model_upper_std_list: list[torch.tensor] = calc_pred_output[4]
        self.acq_fun_vals: np.ndarray = calc_pred_output[5]
        self.fun_arr: torch.tensor = calc_pred_output[6]
        self.samples: torch.tensor = calc_pred_output[8]

        self.point: torch.tensor = calc_pred_output[7]

        self.sdp_acq_func_vals: Optional[tuple[torch.Tensor]] = None

    def add_sdp_acq_func_vals(
            self,
            sdp_acq_func_vals: tuple[torch.Tensor]
    ):
        self.sdp_acq_func_vals = sdp_acq_func_vals


class ModelPredictionContainer:
    def __init__(self):
        self.data: list[ModelPrediction] = []
        self.points: torch.tensor = torch.tensor([])
        self.var_inds: np.ndarray = np.array([])

    def add_data(self, model_prediction: ModelPrediction):
        self.data.append(model_prediction)

        if self.points.numel() == 0:
            self.points = model_prediction.point.unsqueeze(0)
        else:
            self.points = torch.concat([self.points, model_prediction.point.unsqueeze(0)], dim=0)

        self.var_inds = np.append(self.var_inds, model_prediction.var_ind)

    def __getitem__(self, data_ind: int) -> ModelPrediction:
        return self.data[data_ind]

    def locate_data(self, var_ind: int, point: torch.tensor) -> int | None:

        # Can we do without the mix of np and torch here?
        matching_var_inds = torch.tensor(np.equal(var_ind, self.var_inds))

        # NB: Not using any tolerance at the moment, might make this a little unreliable
        no_matching_coordinates_per_point = torch.eq(point, self.points).sum(dim=1)

        n_vars = self.points.shape[1]

        matching_points = no_matching_coordinates_per_point == n_vars

        matching_point_and_var = matching_var_inds * matching_points

        full_match_ind = torch.where(matching_point_and_var)[0]

        if len(full_match_ind) == 1:
            return int(full_match_ind)

        elif full_match_ind.numel() == 0:
            return None

        elif len(full_match_ind) > 1:
            raise RuntimeError("Found more than one matching point.")

        else:
            raise RuntimeError("Unexpected error.")

    def __call__(self, var_ind, point) -> ModelPrediction:
        data_ind = self.locate_data(
            var_ind=var_ind,
            point=point
        )

        if data_ind is None:
            raise ValueError("Point not found.")

        return self.data[data_ind]

    def __contains__(self, point):

        # Just checking if it has it for var_ind = 0, might be sensible to make it a bit more general/stable
        data_ind = self.locate_data(
            var_ind=0,
            point=point
        )

        if data_ind is None:
            return False

        elif type(data_ind) == int:
            return True

        else:
            raise RuntimeError