from typing import Any, Dict, List, TypeVar, Optional, Literal, Union
import time
import json
import os
from veropt.interfaces.simulation import SimulationResult, SimulationRunner
from veropt.interfaces.batch_manager import BatchManager, BatchManagerFactory, ExperimentMode
from veropt.interfaces.result_processing import ResultProcessor, ObjectivesDict
from veropt.interfaces.experiment_utility import (
    ExperimentConfig, OptimiserConfig, ExperimentalState, PathManager, Point
    )
from veropt.interfaces.utility import Config

from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.objective import InterfaceObjective
from veropt.optimiser.constructors import bayesian_optimiser

import torch
from pydantic import BaseModel


SR = TypeVar("SR", bound=SimulationRunner)
ConfigType = TypeVar("ConfigType", bound=Config)


# TODO: Aster, how to handle nan inputs? 
# TODO: Aster, do we implement an option to minimise or maximise in the experiment 
#       (and maximise in VerOpt core by default)?
# TODO: Aster, should VerOpt core be able to read in pre-simulated initial points?
# TODO: Aster, how to ensure that the objectives passed on to the optimiser
#       are in the correct order?    
# TODO: Should the console output be saved when experiment is finished or stopped?
#       - save optimizer object with experiment
# TODO: Aster, here is the list of what experiment wants from the optimiser:
#       - At minimum, what info do I have to pass to the optimiser to initialise it?
#       - Default hyperparameter options for default models
#       - Should Experiment take in the optimiser object in order to change the hyperparameters easily?
#       - How to run a single optimisation step?
#       - How to access objective functions vals and coords in order to sanity check?
#       - If possible, a log of what optimiser does per optimisation step


# TODO: This class could actually take in veropt core structure of the candidates and evaluated points
#       transform + save them, and then load them back in the same structure.
#       Consider if the json files should be temporary.
class ExperimentObjective(InterfaceObjective):
    def __init__(
            self,
            bounds: list[list[float]],
            n_variables: int,
            n_objectives: int,
            suggested_parameters_json: str,
            evaluated_objectives_json: str,
            variable_names: Optional[list[str]] = None,
            objective_names: Optional[list[str]] = None
    ) -> None:

        self.bounds = torch.tensor(bounds)
        self.n_variables = n_variables
        self.n_objectives = n_objectives
        self.suggested_parameters_json = suggested_parameters_json
        self.evaluated_objectives_json = evaluated_objectives_json
        self.variable_names = variable_names
        self.objective_names = objective_names

    def save_candidates(
            self,
            suggested_variables: dict[str, torch.Tensor],
    ) -> None:
        
        suggested_variables_np = {name: value.numpy() for name, value in suggested_variables.items()}

        with open(self.suggested_parameters_json, 'w') as f:
            json.dump(suggested_variables_np, f)

    def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        with open(self.suggested_parameters_json, 'r') as f:
            suggested_variables_np = json.load(f)

        with open(self.evaluated_objectives_json, 'r') as f:
            evaluated_objectives_np = json.load(f)

        suggested_variables = {name: torch.tensor(value) for name, value in suggested_variables_np.items()}
        evaluated_objectives = {name: torch.tensor(value) for name, value in evaluated_objectives_np.items()}

        return suggested_variables, evaluated_objectives


class Experiment:
    def __init__(
        self, 
        simulation_runner: SimulationRunner,
        result_processor: ResultProcessor,
        experiment_config: Union[str, ExperimentConfig],
        optimiser_config: Union[str, OptimiserConfig],
        batch_manager: Optional[BatchManager] = None,
        state: Optional[Union[str, ExperimentalState]] = None
    ) -> None:

        self.experiment_config = ExperimentConfig.load(experiment_config)
        self.optimiser_config = OptimiserConfig.load(optimiser_config)

        self.path_manager = PathManager(self.experiment_config)

        self.state = ExperimentalState.load(state) if state is not None else ExperimentalState.make_fresh_state(
            experiment_name=self.experiment_config.experiment_name,
            experiment_directory=self.path_manager.experiment_directory,
            state_json=self.path_manager.experimental_state_json
        )

        self.simulation_runner = simulation_runner
        self.batch_manager = batch_manager
        self.result_processor = result_processor

        self.initialise_experimental_set_up()

    def _initialise_optimiser(self) -> None:
        self.n_parameters = len(self.experiment_config.parameter_names)
        self.n_objectives = len(self.experiment_config.objective_names)

        # TODO: This is awkward; make not awkward?
        bounds_lower = [bounds[0] for bounds in self.experiment_config.parameter_bounds.values()]
        bounds_upper = [bounds[1] for bounds in self.experiment_config.parameter_bounds.values()]
        self.parameter_bounds = [bounds_lower, bounds_upper]

        objective = ExperimentObjective(
            bounds=self.parameter_bounds,
            n_variables=self.n_parameters,
            n_objectives=self.n_objectives,
            variable_names=self.experiment_config.parameter_names,
            objective_names=self.experiment_config.objective_names,
            suggested_parameters_json=self.path_manager.suggested_parameters_json,
            evaluated_objectives_json=self.path_manager.evaluated_objectives_json
        )

        # TODO: Initialise any optimiser, not just default!
        self.optimiser = bayesian_optimiser(
            n_initial_points=self.optimiser_config.n_initial_points,
            n_bayesian_points=self.optimiser_config.n_bayesian_points,
            n_evaluations_per_step=self.optimiser_config.n_evaluations_per_step,
            objective=objective
        )

    def _initialise_batch_manager(self) -> None:

        self.batch_manager_config = BatchManagerFactory.make_batch_manager_config(
            experiment_mode=self.experiment_config.experiment_mode,
            run_script_filename=self.experiment_config.run_script_filename,
            run_script_root_directory=self.path_manager.run_script_root_directory,
            output_filename=self.experiment_config.output_filename
        )

        self.batch_manager = BatchManagerFactory.make_batch_manager(
            experiment_mode=self.experiment_config.experiment_mode,
            simulation_runner=self.simulation_runner,
            config=self.batch_manager_config
        )

    # TODO: is this redundant?
    def _check_initialisation(self) -> None:

        assert isinstance(self.simulation_runner, SimulationRunner)
        assert isinstance(self.batch_manager, BatchManager)
        assert isinstance(self.result_processor, ResultProcessor)

    def get_parameters_from_optimiser(self) -> dict[int, dict]:

        # TODO: Should this be try with open except?
        with open(self.path_manager.suggested_parameters_json, 'r') as f:
            suggested_parameters = json.load(f)

        # TODO: Is this really necessary and does it work?
        assert suggested_parameters.keys() == self.experiment_config.parameter_names

        dict_of_parameters = {}

        for i in range(self.optimiser.n_evaluations_per_step):

            parameters = {name: value[i] for name, value in suggested_parameters.items()}

            dict_of_parameters[self.state.next_point] = parameters

            new_point = Point(
                parameters=parameters,
                state="Received parameters from core"
            )

            self.state.update(new_point)

        self.state.save_to_json(self.state.state_json)

        return dict_of_parameters

    def save_objectives_to_state(self) -> None:
        ...

    def send_objectives_to_optimiser(
        self,
        objectives: ObjectivesDict
    ) -> None:
    # TODO: Is running an opt step correct to do here?
    #       Or should VerOpt automatically run an opt step when receiving objectives
    #       with a loader method?
        ...

    def _sanity_check(
            self,
            list_of_parameters: list[dict[str, float]],
            results: list[SimulationResult],
            objectives: list[float]
    ) -> None:
        """
        Check if simulations ran with correct params;
        Whether params and objectives match correctly;
        ...
        """
        ...

    def initialise_experimental_set_up(self) -> None:

        self._initialise_optimiser()

        if self.batch_manager is None:
            self._initialise_batch_manager()

        self._check_initialisation()

    def run_experiment_step(self) -> None:

        # TODO: Naming here is poor
        # TODO: Aster, how to evaluate the initial points?
        self.optimiser.run_optimisation_step()
        dict_of_parameters = self.get_parameters_from_optimiser()
        results = self.batch_manager.run_batch(
            dict_of_parameters=dict_of_parameters,
            experimental_state=self.state)
        objectives = self.result_processor.process(results=results)

        self._sanity_check(dict_of_parameters, results, objectives)

        self.save_objectives_to_state(objectives)
        self.send_objectives_to_optimiser(objectives)

    def run_experiment(self) -> None:

        n_iterations = (self.optimiser.n_initial_points + self.optimiser.n_bayesian_points) \
                        // self.optimiser.n_evaluations_per_step

        for i in range(n_iterations):
            self.run_experiment_step()

    def restart_experiment(self) -> None:

        raise NotImplementedError("Restarting an experiment is not implemented yet.")
