import abc
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from veropt import BayesOptimiser


class NormaliserType:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, matrix: torch.Tensor):
        pass

    @abc.abstractmethod
    def transform(self, matrix: torch.Tensor):
        pass

    @abc.abstractmethod
    def inverse_transform(self, matrix: torch.Tensor) -> torch.Tensor:
        pass


class NormaliserZeroMeanUnitVariance(NormaliserType):
    def __init__(self, matrix: torch.Tensor, norm_dim=1):
        self.means = matrix.mean(dim=norm_dim)
        self.variances = matrix.var(dim=norm_dim)

    def transform(self, matrix: torch.Tensor):
        return (matrix - self.means[:, None]) / torch.sqrt(self.variances[:, None])

    def inverse_transform(self, matrix):
        return matrix * torch.sqrt(self.variances[:, None]) + self.means[:, None]


def opacity_for_multidimensional_points(
        var_ind,
        n_params,
        coordinates,
        evaluated_point,
        alpha_min=0.1,
        alpha_max=0.6

):
    distances = []
    index_wo_var_ind = torch.arange(n_params) != var_ind
    for point_no in range(coordinates.shape[1]):
        distances.append(np.linalg.norm(
            evaluated_point[index_wo_var_ind] - coordinates[0, point_no, index_wo_var_ind]))

    distances = torch.tensor(distances)

    norm_distances = ((distances - distances.min()) / distances.max()) / \
                     ((distances - distances.min()) / distances.max()).max()

    norm_proximity = 1 - norm_distances

    alpha_values = (alpha_max - alpha_min) * norm_proximity + alpha_min

    alpha_values[alpha_values.argmax()] = 1.0

    return alpha_values, norm_distances


def get_best_points(
        optimiser: 'BayesOptimiser',
        objs_greater_than: Optional[float | list[float]] = None,
        best_for_obj_ind: Optional[int] = None
):
    n_objs = optimiser.n_objs

    obj_func_coords = optimiser.obj_func_coords
    obj_func_vals = optimiser.obj_func_vals
    obj_weights = optimiser.obj_weights

    assert objs_greater_than is None or best_for_obj_ind is None, "Specifying both options not supported"

    if objs_greater_than is None and best_for_obj_ind is None:

        max_ind = (obj_func_vals * obj_weights).sum(2).argmax()

    elif  objs_greater_than is not None:

        if type(objs_greater_than) == float:

            large_enough_obj_vals = obj_func_vals > objs_greater_than

        elif type(objs_greater_than) == list:

            large_enough_obj_vals = obj_func_vals > torch.tensor(objs_greater_than)

        large_enough_obj_rows = large_enough_obj_vals.sum(dim=2) == n_objs

        if large_enough_obj_rows.max() == 0:
            # Could alternatively raise exception
            return None, None

        filtered_obj_func_vals = obj_func_vals * large_enough_obj_rows.unsqueeze(2)

        max_ind = (filtered_obj_func_vals * obj_weights).sum(2).argmax()

    elif best_for_obj_ind is not None:
        max_ind = obj_func_vals[0, :, best_for_obj_ind].argmax()

    else:
        raise ValueError

    best_coords = obj_func_coords[0, max_ind]
    best_vals = obj_func_vals[0, max_ind]

    return best_coords, best_vals, int(max_ind)


def format_list(unformatted_list):
    formatted_list = "["
    if isinstance(unformatted_list[0], list):
        for it, list_item in enumerate(unformatted_list):
            for number_ind, number in enumerate(list_item):
                if number_ind < len(list_item) - 1:
                    formatted_list += f"{number:.2f}, "
                elif it < len(unformatted_list) - 1:
                    formatted_list += f"{number:.2f}], ["
                else:
                    formatted_list += f"{number:.2f}]"
    else:
        for it, list_item in enumerate(unformatted_list):
            if it < len(unformatted_list) - 1:
                formatted_list += f"{list_item:.2f}, "
            else:
                formatted_list += f"{list_item:.2f}]"

    return formatted_list