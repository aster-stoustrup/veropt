import math
import os
import shutil
import sys
from typing import Self, Union
import json
from pydantic import BaseModel, ConfigDict


def create_directory(path: str) -> None:

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    # TODO: Marta, you okay with deleting this?
    # else:
    #     print(f"Directory already exists: {path}")


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


def _replace_nans_in_json_dict(
        dict_: dict
) -> dict:
    for key, value in dict_.items():
        if isinstance(value, dict):
            dict_[key] = _replace_nans_in_json_dict(
                dict_=value
            )

        elif isinstance(value, float):
            if math.isnan(value):
                dict_[key] = 'NaN'

    return dict_


class Config(BaseModel):

    model_config = ConfigDict(
        ser_json_inf_nan='strings'
    )

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

        with open(config_file, "w") as json_file:

            json_dict = self.model_dump(mode='json')
            json_dict = _replace_nans_in_json_dict(json_dict)  # Necessary because of pydantic bug >:)

            json.dump(
                json_dict,
                json_file,
                indent=2
            )

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
