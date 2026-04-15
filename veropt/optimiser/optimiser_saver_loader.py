import json
import shutil
from typing import Union

from veropt import bayesian_optimiser
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import OptimiserSettings
from veropt.optimiser.saver_loader_utility import SavableClass, TensorsAsListsEncoder
from veropt.optimiser.utility import get_arguments_of_function

# Loaded so it is known by load_optimiser_from_state
from veropt.interfaces.experiment_utility import ExperimentObjective  # noqa: F401

CURRENT_SCHEMA_VERSION = 2


def save_to_json(
        object_to_save: SavableClass,
        file_path: str,
) -> None:
    # TODO: prolly add some path stuff o:)

    save_dict = object_to_save.gather_dicts_to_save()
    save_dict['schema_version'] = CURRENT_SCHEMA_VERSION

    if '.json' in file_path:
        file_path_with_json = file_path
    else:
        file_path_with_json = file_path + '.json'

    with open(file_path_with_json, 'w') as json_file:
        json.dump(
            save_dict,
            json_file,
            cls=TensorsAsListsEncoder,
            indent=2
        )


def load_optimiser_from_state(
        file_name: str
) -> 'BayesianOptimiser':

    with open(file_name, 'r') as json_file:
        saved_dict = json.load(json_file)

    schema_version = saved_dict.get('schema_version', 1)

    if schema_version < CURRENT_SCHEMA_VERSION:
        allow_updates = (
            saved_dict
            .get('optimiser', {})
            .get('settings', {})
            .get('allow_automatic_json_updates', False)
        )

        if allow_updates:
            migrate_json(file_name)
            with open(file_name, 'r') as json_file:
                saved_dict = json.load(json_file)
        else:
            raise RuntimeError(
                f"The optimiser JSON at '{file_name}' uses schema version {schema_version}, "
                f"but the current schema version is {CURRENT_SCHEMA_VERSION}. "
                f"To update the file, either:\n"
                f"  1. Run: from veropt.optimiser.optimiser_saver_loader import migrate_json\n"
                f"          migrate_json('{file_name}')\n"
                f"  2. Set 'allow_automatic_json_updates=True' in your optimiser config — future "
                f"saves will include this flag and auto-migrate on load."
            )

    return BayesianOptimiser.from_saved_state(saved_dict['optimiser'])


def _migrate_v1_to_v2(saved_dict: dict) -> dict:
    """Move noise fields (noise, noise_lower_bound, train_noise) from inside each kernel's
    'settings' dict to the top level of the kernel's state dict."""

    _NOISE_KEYS = frozenset({'noise', 'noise_lower_bound', 'train_noise'})

    try:
        model_dicts = (
            saved_dict
            ['optimiser']
            ['predictor']
            ['state']
            ['model']
            ['state']
            ['model_dicts']
        )
    except KeyError as missing_key:
        raise RuntimeError(
            f"Could not migrate JSON: unexpected structure. Missing key: {missing_key}. "
            "This file may already be in a non-standard format."
        ) from missing_key

    for model_key, model_dict in model_dicts.items():
        kernel_settings = model_dict['state'].get('settings', {})
        noise_values = {key: value for key, value in kernel_settings.items() if key in _NOISE_KEYS}
        clean_settings = {key: value for key, value in kernel_settings.items() if key not in _NOISE_KEYS}

        model_dicts[model_key]['state']['settings'] = clean_settings
        model_dicts[model_key]['state'].update(noise_values)

    saved_dict['schema_version'] = CURRENT_SCHEMA_VERSION

    return saved_dict


def migrate_json(file_path: str) -> None:
    """Migrate a saved optimiser JSON file to the current schema version.

    A backup is written to <file_path>.bak before the original is modified.
    The backup is written and flushed before touching the original, so both
    files are never in a corrupt state simultaneously.
    """

    file_path_with_json = file_path if '.json' in file_path else file_path + '.json'
    backup_path = file_path_with_json + '.bak'

    with open(file_path_with_json, 'r') as json_file:
        saved_dict = json.load(json_file)

    schema_version = saved_dict.get('schema_version', 1)

    if schema_version >= CURRENT_SCHEMA_VERSION:
        print(f"File '{file_path_with_json}' is already at schema version {schema_version}. No migration needed.")
        return

    # Write backup BEFORE modifying anything — use shutil.copy2 to preserve metadata
    shutil.copy2(file_path_with_json, backup_path)

    if schema_version < 2:
        saved_dict = _migrate_v1_to_v2(saved_dict)

    with open(file_path_with_json, 'w') as json_file:
        json.dump(saved_dict, json_file, cls=TensorsAsListsEncoder, indent=2)

    print(
        f"Migration complete: schema v{schema_version} → v{CURRENT_SCHEMA_VERSION}. "
        f"Backup saved at '{backup_path}'."
    )


def load_optimiser_from_settings(
        file_name: str,
        objective: Union[InterfaceObjective, CallableObjective],
) -> 'BayesianOptimiser':

    with open(file_name, 'r') as json_file:
        settings_dict = json.load(json_file)

    required_arguments = get_arguments_of_function(
        function=bayesian_optimiser,
        argument_type='required',
        excluded_arguments=['objective']
    )

    for required_parameter in required_arguments:
        assert required_parameter in settings_dict, (
            f"The top level of an optimiser settings file must contain (at least) {required_arguments} "
            f"but got {list(settings_dict.keys())}"
        )

    all_arguments = get_arguments_of_function(
        function=bayesian_optimiser,
        excluded_arguments=['objective', 'kwargs']
    )

    all_arguments += get_arguments_of_function(
        function=OptimiserSettings.__init__,
        excluded_arguments=['self', 'n_objectives'] + all_arguments
    )

    for key in settings_dict.keys():
        assert key in all_arguments, f"Key '{key}' not recognised. Must be one of {all_arguments}."

    return bayesian_optimiser(
        objective=objective,
        **settings_dict
    )
