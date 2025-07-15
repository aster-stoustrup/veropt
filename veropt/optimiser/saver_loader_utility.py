import abc
from dataclasses import asdict, dataclass, fields
from inspect import isabstract
from json import JSONEncoder
from typing import Self

import torch
from torch.utils.data import Dataset


class SavableClass(metaclass=abc.ABCMeta):

    name: str = 'meta'

    @abc.abstractmethod
    def gather_dicts_to_save(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        pass


@dataclass
class SavableDataClass(SavableClass):

    def gather_dicts_to_save(self) -> dict:
        return asdict(self)

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        expected_fields = [field.name for field in fields(cls)]

        for key in saved_state.keys():
            assert key in expected_fields, f"Field '{key}' from saved state not expected for dataclass {cls.__name__}"

        for expected_field in expected_fields:
            assert expected_field in saved_state, f"Field '{expected_field}' not found in saved state"

        return cls(
            **saved_state
        )


class EmptyDataClass(SavableDataClass):
    pass


def get_all_subclasses[T: type](
        cls: T
) -> list[T]:

    return cls.__subclasses__() + (
        [subclass for class_ in cls.__subclasses__() for subclass in get_all_subclasses(class_)]
    )


def rehydrate_object[S: SavableClass](
        superclass: type[S],
        name: str,
        saved_state: dict,
) -> S:

    subclasses = get_all_subclasses(superclass)

    for subclass in [superclass] + subclasses:
        if not isabstract(subclass):
            if subclass.name == name:
                return subclass.from_saved_state(
                    saved_state=saved_state
                )

    else:
        raise ValueError(f"Unknown subclass of {superclass.__name__}: '{name}'")


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
