import json
from json import JSONEncoder
from typing import Optional, TypeVar

import torch
from torch.utils.data import Dataset

from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.utility import SavableClass


# TODO: Implement
#   - Need to be able to rebuild optimiser object
#   - Probably need some jsons and saving torch models


# TODO:
#   - Make method that writes json for all save-able classes?!
#       - I guess we'd like to collect it all into one json though?


class TensorsAsListsEncoder(JSONEncoder, Dataset):

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

        return super(TensorsAsListsEncoder, self).default(dict_item)


def save_to_json(
        object_to_save: SavableClass,
        file_name: str,
) -> None:
    # TODO: prolly add some path stuff o:)

    save_dict = object_to_save.gather_dicts_to_save()

    with open(f'{file_name}.json', 'w') as json_file:
        json.dump(save_dict, json_file, cls=TensorsAsListsEncoder)


def load_optimiser_from_json(
        file_name: str
) -> 'BayesianOptimiser':

    with open(f'{file_name}.json', 'r') as json_file:
        saved_dict = json.load(json_file)

    print("this is a breakpoint >:)")

    return BayesianOptimiser.from_saved_state(saved_dict['optimiser'])


def get_all_subclasses[T](
        cls: T
) -> list[T]:

    return cls.__subclasses__() + (
        [subclass for class_ in cls.__subclasses__() for subclass in get_all_subclasses(class_)]
    )


# TODO: Relation to constructors...?

# We're not showing that it's the superclass in type[T] but a subclass in the return
#   - Not sure if this is a problem or not?
T = TypeVar('T', bound=SavableClass)

def rehydrate_object(
        superclass: type[T],
        name: str,
        saved_state: dict,
        subclasses: Optional[list[T]] = None
) -> T:

    subclasses = subclasses or get_all_subclasses(superclass)

    for subclass in subclasses:
        if subclass.name == name:
            return subclass.from_saved_state(
                saved_state=saved_state
            )

    else:
        raise ValueError(f"Unknown subclass of {superclass.__name__}: '{name}'")
