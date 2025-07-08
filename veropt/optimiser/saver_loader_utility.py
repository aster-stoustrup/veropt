import abc
from dataclasses import asdict, dataclass, fields
from typing import Self, TypeVar
from inspect import isabstract


class SavableClass(metaclass=abc.ABCMeta):

    name: str

    @abc.abstractmethod
    def gather_dicts_to_save(self) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def from_saved_state(cls, saved_state: dict) -> Self:
        pass


@dataclass
class SavableDataClass(SavableClass):

    def gather_dicts_to_save(self) -> dict:
        return asdict(self)

    @classmethod
    def from_saved_state(cls, saved_state: dict) -> Self:

        expected_fields = [field.name for field in fields(cls)]

        for key in saved_state.keys():
            assert key in expected_fields, f"Field '{key}' from saved state not expected for dataclass {cls.__name__}"

        for expected_field in expected_fields:
            assert expected_field in saved_state, f"Field '{expected_field}' not found in saved state"

        return cls(
            **saved_state
        )


T = TypeVar('T', bound=type)


def get_all_subclasses(
        cls: T
) -> list[T]:

    return cls.__subclasses__() + (
        [subclass for class_ in cls.__subclasses__() for subclass in get_all_subclasses(class_)]
    )


SavableSettings = TypeVar('SavableSettings', bound=SavableDataClass)


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
