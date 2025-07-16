import os
import shutil
import sys
from typing import Self, Union

from pydantic import BaseModel


def create_directory(path: str) -> None:

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    else:
        print(f"Directory already exists: {path}")


def copy_files(
        source_directory: str,
        destination_directory: str
) -> None:

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory, exist_ok=True)

    for file_name in os.listdir(source_directory):
        source_file = os.path.join(source_directory, file_name)
        destination_file = os.path.join(destination_directory, file_name)

        if os.path.isfile(source_file):
            if not os.path.exists(destination_file):
                shutil.copy(source_file, destination_directory)
                print(f"Copied {source_file} to {destination_directory}")
            else:
                print(f"File already exists: {destination_file}")

        else:
            print(f"Skipping non-file: {source_file}")


class Config(BaseModel):

    @classmethod
    def load(
            cls,
            source: Union[str, Self]
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
            print(f"While reading {path}: {e}")
            sys.exit()

        return loaded_class
