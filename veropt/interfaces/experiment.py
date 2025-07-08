from typing import TypeVar, Optional, Union, Generic, Self
import json
from veropt.interfaces.simulation import SimulationResult, SimulationRunner
from veropt.interfaces.batch_manager import BatchManager, batch_manager
from veropt.interfaces.result_processing import ResultProcessor, ObjectivesDict
from veropt.interfaces.experiment_utility import (
    ExperimentConfig, OptimiserConfig, ExperimentalState, PathManager, Point
    )
from veropt.interfaces.utility import Config

from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.objective import InterfaceObjective
from veropt.optimiser.constructors import bayesian_optimiser

import torch

torch.set_default_dtype(torch.float64)


SR = TypeVar("SR", bound=SimulationRunner)
RP = TypeVar("RP", bound=ResultProcessor)
BM = TypeVar("BM", bound=BatchManager)


class ExperimentObjective(InterfaceObjective):
    def __init__(
            self,
            bounds_lower: list[float],
            bounds_upper: list[float],
            n_variables: int,
            n_objectives: int,
            variable_names: list[str],
            objective_names: list[str],
            suggested_parameters_json: str,
            evaluated_objectives_json: str
    ):

        self.suggested_parameters_json = suggested_parameters_json
        self.evaluated_objectives_json = evaluated_objectives_json

        super().__init__(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            n_variables=n_variables,
            n_objectives=n_objectives,
            variable_names=variable_names,
            objective_names=objective_names
            )

    def save_candidates(
            self,
            suggested_variables: dict[str, torch.Tensor],
    ) -> None:

        print(suggested_variables)
        
        suggested_variables_np = {name: value.tolist() for name, value in suggested_variables.items()}

        with open(self.suggested_parameters_json, 'w') as f:
            json.dump(suggested_variables_np, f)

    def load_evaluated_points(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        with open(self.suggested_parameters_json, 'r') as f:
            suggested_variables_np = json.load(f)

        with open(self.evaluated_objectives_json, 'r') as f:
            evaluated_objectives_np = json.load(f)

        suggested_variables = {name: torch.tensor(value) for name, value in suggested_variables_np.items()}
        evaluated_objectives = {name: torch.tensor(value) for name, value in evaluated_objectives_np.items()}

        print(suggested_variables)
        print(evaluated_objectives)

        return suggested_variables, evaluated_objectives
    
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:
        pass


class Experiment(Generic[SR, RP, BM]):
    def __init__(
            self, 
            simulation_runner: SR,
            result_processor: RP,
            experiment_config: Union[str, ExperimentConfig],
            optimiser_config: Union[str, OptimiserConfig],
            batch_manager: Optional[BM] = None,
            state: Optional[Union[str, ExperimentalState]] = None
    ):

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

        self.n_parameters = len(self.experiment_config.parameter_names)
        self.n_objectives = len(self.experiment_config.objective_names)

        self.initialise_experimental_set_up()

    def _initialise_optimiser(self) -> None:

        bounds_lower = [self.experiment_config.parameter_bounds[name][0] \
                        for name in self.experiment_config.parameter_names]
        bounds_upper = [self.experiment_config.parameter_bounds[name][1] \
                        for name in self.experiment_config.parameter_names]

        objective = ExperimentObjective(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
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

        self.batch_manager = batch_manager(
            experiment_mode=self.experiment_config.experiment_mode,
            simulation_runner=self.simulation_runner,
            run_script_filename=self.experiment_config.run_script_filename,
            run_script_root_directory=self.path_manager.run_script_root_directory,
            results_directory=self.path_manager.results_directory,
            output_filename=self.experiment_config.output_filename,
            check_job_status_sleep_time=60,  # TODO: put in config
            )
        
    def _initialise_objective_jsons(self) -> None:
        initial_parameter_dict = {name: [] for name in self.experiment_config.parameter_names}
        initial_objectives_dict = {name: [] for name in self.experiment_config.objective_names}

        with open(self.path_manager.suggested_parameters_json, "w") as file:
            json.dump(initial_parameter_dict, file)

        with open(self.path_manager.evaluated_objectives_json, "w") as file:
            json.dump(initial_objectives_dict, file)

    def _check_initialisation(self) -> None:

        assert isinstance(self.simulation_runner, SimulationRunner), "simulation_runner must be a SimulationRunner"
        assert isinstance(self.batch_manager, BatchManager), "batch_manager must be a BatchManager"
        assert isinstance(self.result_processor, ResultProcessor), "result_processor must be a ResultProcessor"

    def get_parameters_from_optimiser(self) -> dict[int, dict]:

        # TODO: Should this be try with open except?
        with open(self.path_manager.suggested_parameters_json, 'r') as f:
            suggested_parameters = json.load(f)

        assert [key for key in suggested_parameters.keys()] == self.experiment_config.parameter_names

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

    def save_objectives_to_state(
            self,
            dict_of_objectives: ObjectivesDict) -> None:
        
        # TODO: consider naming
        for i, objective_values in dict_of_objectives.items():
            self.state.points[i].objective_values = objective_values

        self.state.save_to_json(self.state.state_json)

    def send_objectives_to_optimiser(
            self,
            dict_of_parameters: dict[int, dict],
            dict_of_objectives: ObjectivesDict
    ) -> None:
        
        evaluated_objectives = {name: [dict_of_objectives[i][name] for i in dict_of_parameters.keys()] \
                                for name in self.experiment_config.objective_names}
        
        print(evaluated_objectives)
        
        with open(self.path_manager.evaluated_objectives_json, "w") as file:
            json.dump(evaluated_objectives, file)

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
        self._initialise_objective_jsons()

        if self.batch_manager is None:
            self._initialise_batch_manager()

        self._check_initialisation()

    def run_experiment_step(self) -> None:

        self.optimiser.run_optimisation_step()
        dict_of_parameters = self.get_parameters_from_optimiser()
        results = self.batch_manager.run_batch(
            dict_of_parameters=dict_of_parameters,
            experimental_state=self.state)
        dict_of_objectives = self.result_processor.process(results=results)

        # self._sanity_check(dict_of_parameters, results, objectives)

        self.save_objectives_to_state(dict_of_objectives=dict_of_objectives)
        self.send_objectives_to_optimiser(
            dict_of_parameters=dict_of_parameters,
            dict_of_objectives=dict_of_objectives)

    def run_experiment(self) -> None:

        n_iterations = (self.optimiser.n_initial_points + self.optimiser.n_bayesian_points) \
                        // self.optimiser.n_evaluations_per_step

        for i in range(n_iterations):
            self.run_experiment_step()

    def restart_experiment(self) -> None:

        raise NotImplementedError("Restarting an experiment is not implemented yet.")

