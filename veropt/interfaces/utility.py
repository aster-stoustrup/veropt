from pydantic import BaseModel
import json
import os
import sys
from typing import Self, Union, Optional, Type, TypeVar
from abc import abstractmethod


class Config(BaseModel):

    @classmethod
    def load(
        cls,
        source: Optional[Union[str, Self]] = None
    ) -> Self:

        if isinstance(source, str):
            return cls.load_from_json(source)
        elif isinstance(source, cls):
            return source
        else:
            raise ValueError(f"Invalid source type: {type(source)}. Expected str or {cls.__name__} instance.")

    def save_to_json(
        self,
        config_file: str
    ) -> None:

        with open(config_file, "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load_from_json(
        cls,
        path: str
    ) -> Self:

        try:
            with open(path, "r") as f:
                loaded_class = cls.model_validate_json(f.read())
        except Exception as e:
            print(f"While reading {path}:{e}")
            sys.exit(1)

        return loaded_class
