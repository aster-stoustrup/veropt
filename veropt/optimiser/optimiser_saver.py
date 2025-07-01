import abc
import json
from json import JSONEncoder

import torch
from torch.utils.data import Dataset


# TODO: Implement
#   - Need to be able to rebuild optimiser object
#   - Probably need some jsons and saving torch models


# TODO:
#   - Make method that writes json for all save-able classes?!
#       - I guess we'd like to collect it all into one json though?


class SavableClass:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gather_dicts_to_save(self) -> dict:
        pass


class SavableDataClass(SavableClass):
    def gather_dicts_to_save(self) -> dict:
        return self.__dict__


class TensorsAsLists(JSONEncoder, Dataset):

    def default(
            self,
            dict_item
    ):

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

        return super(TensorsAsLists, self).default(dict_item)


def save_to_json(
        object_to_save: SavableClass,
        file_name: str,
) -> None:
    # TODO: prolly add some path stuff o:)

    save_dict = object_to_save.gather_dicts_to_save()

    with open(f'{file_name}.json', 'w') as json_file:
        json.dump(save_dict, json_file, cls=TensorsAsLists)


def load_optimiser_from_json(
        file_name: str
):
    with open(f'{file_name}.json', 'r') as json_file:
        saved_dict = json.load(json_file)

    print("this is a breakpoint >:)")
