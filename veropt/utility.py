import abc
import time
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from veropt import BayesOptimiser
    from veropt import AcqFunction


# TODO: Move
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


def np_input_support(func):
    def wrapper(self, data_in):

        if type(data_in) == np.ndarray:
            new_data_in = torch.tensor(data_in)
            need_to_convert_back = True
        else:
            new_data_in = data_in
            need_to_convert_back = False

        data_out = func(self, new_data_in)

        if need_to_convert_back:
            return data_out.detach().numpy()
        else:
            return data_out

    return wrapper


# TODO: Move
class NormaliserZeroMeanUnitVariance(NormaliserType):
    def __init__(self, matrix: torch.Tensor, norm_dim=1):
        self.means = matrix.mean(dim=norm_dim)
        self.variances = matrix.var(dim=norm_dim)

    @np_input_support
    def transform(self, matrix: torch.Tensor):
        return (matrix - self.means[:, None]) / torch.sqrt(self.variances[:, None])

    @np_input_support
    def inverse_transform(self, matrix: torch.Tensor):
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
            evaluated_point[index_wo_var_ind] - coordinates[0, point_no, index_wo_var_ind])
        )

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


# TODO: move or delete?
def speed_test_acq_func(
        acq_func: 'AcqFunction'
):

    time_start = time.time()

    n_acq_func_samples = 10
    n_params = acq_func.bounds.shape[1]

    random_coordinates = (
            (acq_func.bounds[1] - acq_func.bounds[0]) * torch.rand(n_acq_func_samples, n_params)
            + acq_func.bounds[0]
    )

    random_coordinates = random_coordinates.reshape(n_acq_func_samples, 1, n_params)

    samples = acq_func.function(random_coordinates)

    time_stop = time.time()
    time_taken_batched_a = time_stop - time_start


    time_start = time.time()

    random_coordinates = (
            (acq_func.bounds[1] - acq_func.bounds[0]) * torch.rand(n_acq_func_samples, n_params)
            + acq_func.bounds[0]
    )

    random_coordinates = random_coordinates.reshape(n_acq_func_samples, 1, n_params)

    for coordinate_ind in range(n_acq_func_samples):
        samples = acq_func.function(random_coordinates[coordinate_ind:coordinate_ind+1, :, :])

    time_stop = time.time()
    time_taken_sequential = time_stop - time_start


    time_start = time.time()

    random_coordinates = (
            (acq_func.bounds[1] - acq_func.bounds[0]) * torch.rand(n_acq_func_samples, n_params)
            + acq_func.bounds[0]
    )

    random_coordinates = random_coordinates.reshape(n_acq_func_samples, 1, n_params)

    for coordinate_ind in range(n_acq_func_samples):
        samples = acq_func.function(random_coordinates[coordinate_ind:coordinate_ind+1, :, :])
        samples.detach().numpy()

    time_stop = time.time()
    time_taken_sequential_np = time_stop - time_start


    print(f"time batched A: {time_taken_batched_a}")
    print(f"time sequential: {time_taken_sequential}")
    print(f"time sequential np: {time_taken_sequential_np}")

