import json
from typing import Union

from veropt import bayesian_optimiser
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import OptimiserSettings
from veropt.optimiser.saver_loader_utility import SavableClass, TensorsAsListsEncoder
from veropt.optimiser.utility import get_arguments_of_function


def save_to_json(
        object_to_save: SavableClass,
        file_name: str,
) -> None:
    # TODO: prolly add some path stuff o:)

    save_dict = object_to_save.gather_dicts_to_save()

    # TODO: Check if file_name has .json ending or not, add if not

    with open(f'{file_name}.json', 'w') as json_file:
        json.dump(save_dict, json_file, cls=TensorsAsListsEncoder)


def load_optimiser_from_state(
        file_name: str
) -> 'BayesianOptimiser':

    with open(file_name, 'r') as json_file:
        saved_dict = json.load(json_file)

    return BayesianOptimiser.from_saved_state(saved_dict['optimiser'])


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
