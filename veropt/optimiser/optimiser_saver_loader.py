import json
from json import JSONEncoder
from typing import Union

import torch
from torch.utils.data import Dataset

from veropt.optimiser import bayesian_optimiser
from veropt.optimiser.objective import CallableObjective, InterfaceObjective, Objective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.saver_loader_utility import SavableClass


class TensorsAsListsEncoder(JSONEncoder, Dataset):

    def default[T](
            self,
            dict_item: T
    ) -> T | list | str:

        if isinstance(dict_item, torch.Tensor):

            converted_tensor = dict_item.detach().tolist()

            if type(converted_tensor) is list:
                converted_tensor = [
                    str(item) if item in [float('inf'), float('-inf')] else item for item in converted_tensor
                ]

            elif type(converted_tensor) is float:
                converted_tensor = (
                    str(converted_tensor) if converted_tensor in [float('inf'), float('-inf')] else converted_tensor
                )

            return converted_tensor

        return super(TensorsAsListsEncoder, self).default(dict_item)


def save_to_json(
        object_to_save: SavableClass,
        file_name: str,
) -> None:
    # TODO: prolly add some path stuff o:)

    save_dict = object_to_save.gather_dicts_to_save()

    with open(f'{file_name}.json', 'w') as json_file:
        json.dump(save_dict, json_file, cls=TensorsAsListsEncoder)


def load_optimiser_from_state(
        file_name: str
) -> 'BayesianOptimiser':

    with open(f'{file_name}.json', 'r') as json_file:
        saved_dict = json.load(json_file)

    return BayesianOptimiser.from_saved_state(saved_dict['optimiser'])


def load_optimiser_from_settings(
        file_name: str,
        objective: Union[InterfaceObjective, CallableObjective],
) -> 'BayesianOptimiser':

    with open(f'{file_name}.json', 'r') as json_file:
        saved_dict = json.load(json_file)

    return bayesian_optimiser(
        objective=objective,
        **saved_dict
    )
