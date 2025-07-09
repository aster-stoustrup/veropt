import json
from typing import Union

from veropt.optimiser import bayesian_optimiser
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.saver_loader_utility import SavableClass, TensorsAsListsEncoder


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

    # TODO: Add some checks to see if the file is okay
    #   - Can partially rely on the nice errors from the constructor
    #   - But need to check...
    #           - presence of mandatory arguments
    #           - that keyword arguments are correct for the top-level constructor

    with open(f'{file_name}.json', 'r') as json_file:
        saved_dict = json.load(json_file)

    return bayesian_optimiser(
        objective=objective,
        **saved_dict
    )
