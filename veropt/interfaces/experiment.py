import os.path
from typing import Optional, Union, Self
import json

from veropt.interfaces.simulation import SimulationRunner
from veropt.interfaces.batch_manager import make_batch_manager, DirectBatchManager, \
    SubmitBatchManager
from veropt.interfaces.result_processing import ResultProcessor, ObjectivesDict
from veropt.interfaces.experiment_utility import (
    ExperimentConfig, ExperimentMode, ExperimentalState, PathManager, Point
)
from veropt.optimiser.objective import InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_saver_loader import (
    bayesian_optimiser, load_optimiser_from_settings, load_optimiser_from_state, save_to_json
)

import torch
import numpy as np

torch.set_default_dtype(torch.float64)


def _mask_nans(
        dict_of_objectives: ObjectivesDict,
        experimental_state: ExperimentalState
) -> ObjectivesDict:  # TODO: Remove when veropt core supports nan imputs

    current_minima: dict[str, float] = {}
    current_stds: dict[str, float] = {}
    first_new_point = next(iter(dict_of_objectives.values()))

    assert experimental_state.points, "To clear nans, there must be at least one point saved to state."

    for objective_name in first_new_point.keys():
        objective_values = []

        for i in range(experimental_state.next_point):
            if experimental_state.points[i].objective_values is not None:  # I check if it is None right here
                objective_values.append(experimental_state.points[i].objective_values[objective_name])  # type: ignore
            else:
                continue

        assert objective_values, f'No objective values found for objective "{objective_name}".'

        current_minima[objective_name] = np.nanmin(objective_values)  # type: ignore[arg-type]  # Type checked above
        current_stds[objective_name] = np.nanstd(objective_values).astype(float)  # type: ignore[arg-type]

        assert not np.isnan(current_minima[objective_name]), (
            f'All objective values are nans for objective "{objective_name}".'
        )

    for i, objectives in dict_of_objectives.items():
        dict_of_objectives[i] = {
            name: value if not np.isnan(value) else current_minima[name] - 2 * current_stds[name]
            for name, value in objectives.items()
        }

    return dict_of_objectives


class ExperimentObjective(InterfaceObjective):

    name = "experiment_objective"

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

        return suggested_variables, evaluated_objectives

    def gather_dicts_to_save(self) -> dict:
        saved_state = super().gather_dicts_to_save()
        saved_state['state']['suggested_parameters_json'] = self.suggested_parameters_json
        saved_state['state']['evaluated_objectives_json'] = self.evaluated_objectives_json
        return saved_state

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        bounds_lower = saved_state["bounds"][0]
        bounds_upper = saved_state["bounds"][1]

        return cls(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            n_variables=saved_state["n_variables"],
            n_objectives=saved_state["n_objectives"],
            variable_names=saved_state["variable_names"],
            objective_names=saved_state["objective_names"],
            suggested_parameters_json=saved_state["suggested_parameters_json"],
            evaluated_objectives_json=saved_state["evaluated_objectives_json"]
        )


class Experiment:
    def __init__(
            self,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: ExperimentConfig,
            optimiser: BayesianOptimiser,
            path_manager: PathManager,
            batch_manager: Union[DirectBatchManager, SubmitBatchManager],
            state: ExperimentalState
    ):
        self.experiment_config = experiment_config
        self.path_manager = path_manager

        self.simulation_runner = simulation_runner
        self.batch_manager = batch_manager
        self.result_processor = result_processor

        self.state = state
        self.optimiser = optimiser

        self.n_parameters = len(self.experiment_config.parameter_names)
        self.n_objectives = len(self.result_processor.objective_names)

    @classmethod
    def from_the_beginning(
            cls,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: Union[str, ExperimentConfig],
            optimiser_config: Union[str, dict],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None
    ) -> Self:

        experiment_config = ExperimentConfig.load(experiment_config)
        path_manager = PathManager(experiment_config)

        state = ExperimentalState.make_fresh_state(
            experiment_name=experiment_config.experiment_name,
            experiment_directory=path_manager.experiment_directory,
            state_json=path_manager.experimental_state_json
        )

        state_json_path = path_manager.experimental_state_json

        if os.path.exists(state_json_path):
            raise RuntimeError(
                f"Experimental state exists at {state_json_path}. Please clear all files from previous run,"
                f"unless you want to continue that run. (In that case, use .continue_if_possible instead of"
                f".from_the_beginning.)"
            )

        n_parameters = len(experiment_config.parameter_names)
        n_objectives = len(result_processor.objective_names)

        optimiser = cls._make_fresh_optimiser(
            n_parameters=n_parameters,
            n_objectives=n_objectives,
            experiment_config=experiment_config,
            result_processor=result_processor,
            path_manager=path_manager,
            optimiser_config=optimiser_config,
        )

        batch_manager = cls._make_fresh_batch_manager(
            experiment_config=experiment_config,
            simulation_runner=simulation_runner,
            path_manager=path_manager,
            batch_manager_class=batch_manager_class
        )

        experiment = cls(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser=optimiser,
            path_manager=path_manager,
            batch_manager=batch_manager,
            state=state
        )

        experiment._initialise_objective_jsons()

        return experiment

    @classmethod
    def _continue_existing(
            cls,
            state_path: str,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: Union[str, ExperimentConfig],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None
    ) -> Self:

        experiment_config = ExperimentConfig.load(experiment_config)
        path_manager = PathManager(experiment_config)

        state = ExperimentalState.load(state_path)

        batch_manager = cls._make_fresh_batch_manager(
            experiment_config=experiment_config,
            simulation_runner=simulation_runner,
            path_manager=path_manager,
            batch_manager_class=batch_manager_class
        )

        optimiser = load_optimiser_from_state(
            file_name=path_manager.optimiser_state_json
        )

        return cls(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser=optimiser,
            path_manager=path_manager,
            batch_manager=batch_manager,
            state=state
        )

    @classmethod
    def continue_if_possible(
            cls,
            simulation_runner: SimulationRunner,
            result_processor: ResultProcessor,
            experiment_config: Union[str, ExperimentConfig],
            optimiser_config: Union[str, dict],
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]] = None
    ) -> Self:

        experiment_config = ExperimentConfig.load(experiment_config)
        path_manager = PathManager(experiment_config)

        state_json_path = path_manager.experimental_state_json

        if os.path.exists(state_json_path):
            return cls._continue_existing(
                state_path=state_json_path,
                simulation_runner=simulation_runner,
                result_processor=result_processor,
                experiment_config=experiment_config,
                batch_manager_class=batch_manager_class
            )
        else:
            return cls.from_the_beginning(
                simulation_runner=simulation_runner,
                result_processor=result_processor,
                experiment_config=experiment_config,
                optimiser_config=optimiser_config,
                batch_manager_class=batch_manager_class
            )

    @staticmethod
    def _make_fresh_optimiser(
            n_parameters: int,
            n_objectives: int,
            experiment_config: ExperimentConfig,
            result_processor: ResultProcessor,
            path_manager: PathManager,
            optimiser_config: Union[str, dict]
    ) -> BayesianOptimiser:

        bounds_lower = [experiment_config.parameter_bounds[name][0]
                        for name in experiment_config.parameter_names]
        bounds_upper = [experiment_config.parameter_bounds[name][1]
                        for name in experiment_config.parameter_names]

        objective = ExperimentObjective(
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            n_variables=n_parameters,
            n_objectives=n_objectives,
            variable_names=experiment_config.parameter_names,
            objective_names=result_processor.objective_names,
            suggested_parameters_json=path_manager.suggested_parameters_json,
            evaluated_objectives_json=path_manager.evaluated_objectives_json
        )

        if isinstance(optimiser_config, str):
            return load_optimiser_from_settings(
                file_name=optimiser_config,
                objective=objective
            )

        elif isinstance(optimiser_config, dict):
            return bayesian_optimiser(
                objective=objective,
                **optimiser_config
            )

        else:
            raise ValueError("optimiser_config must be a valid dictionary or a path to a valid configuration file")

    @staticmethod
    def _make_fresh_batch_manager(
            experiment_config: ExperimentConfig,
            simulation_runner: SimulationRunner,
            path_manager: PathManager,
            batch_manager_class: Optional[Union[type[DirectBatchManager], type[SubmitBatchManager]]]
    ) -> Union[DirectBatchManager, SubmitBatchManager]:

        # TODO: Refactor this (consider how to set up this little guy)
        #   - Fixed this so it works but isn't general
        #   - Probably need a constructor that makes the experiment itself...?
        #   - Or actually, might mostly need to make an interface to the batch_manager where it can take
        #     some of the things it is being bound to here?
        #       - Then it can be initialised on its own but can be linked to internal things here or something

        return make_batch_manager(
            experiment_mode=experiment_config.experiment_mode.name,  # type: ignore[arg-type]  # Silly mypy
            simulation_runner=simulation_runner,
            run_script_filename=experiment_config.run_script_filename,
            run_script_root_directory=path_manager.run_script_root_directory,
            results_directory=path_manager.results_directory,
            output_filename=experiment_config.output_filename,
            check_job_status_frequency=60,  # TODO: put in server config,
            batch_manager_class=batch_manager_class
        )

    def _initialise_objective_jsons(self) -> None:

        initial_parameter_dict: dict[str, list] = {name: [] for name in self.experiment_config.parameter_names}
        initial_objectives_dict: dict[str, list] = {name: [] for name in self.result_processor.objective_names}

        with open(self.path_manager.suggested_parameters_json, "w") as f:
            json.dump(initial_parameter_dict, f)

        with open(self.path_manager.evaluated_objectives_json, "w") as f:
            json.dump(initial_objectives_dict, f)

    def get_parameters_from_optimiser(self) -> dict[int, dict]:

        with open(self.path_manager.suggested_parameters_json, 'r') as f:
            suggested_parameters = json.load(f)

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
            dict_of_objectives: ObjectivesDict
    ) -> None:

        for i, objective_values in dict_of_objectives.items():
            self.state.points[i].objective_values = objective_values  # type: ignore[assignment]  # mypy silliness

        self.state.save_to_json(self.state.state_json)

    def send_objectives_to_optimiser(
            self,
            dict_of_parameters: dict[int, dict],
            dict_of_objectives: ObjectivesDict
    ) -> None:

        dict_of_objectives = _mask_nans(
            dict_of_objectives=dict_of_objectives,
            experimental_state=self.state
        )  # TODO: Remove when veropt core supports nan imputs

        evaluated_objectives = {name: [dict_of_objectives[i][name] for i in dict_of_parameters.keys()]
                                for name in self.result_processor.objective_names}

        with open(self.path_manager.evaluated_objectives_json, "w") as f:
            json.dump(evaluated_objectives, f)

    def _save_optimiser(self) -> None:

        save_to_json(
            object_to_save=self.optimiser,
            file_path=self.path_manager.optimiser_state_json
        )

    def run_experiment_step_direct(self) -> None:

        assert issubclass(type(self.batch_manager), DirectBatchManager), (
            "Batch manager must be subclassing DirectBatchManager to call this method."
        )
        self.optimiser.run_optimisation_step()

        dict_of_parameters = self.get_parameters_from_optimiser()

        results = self.batch_manager.run_batch(  # type: ignore[union-attr]  # Checked above
            dict_of_parameters=dict_of_parameters,
            experimental_state=self.state
        )

        dict_of_objectives = self.result_processor.process(results=results)

        self.save_objectives_to_state(dict_of_objectives=dict_of_objectives)
        self.send_objectives_to_optimiser(
            dict_of_parameters=dict_of_parameters,
            dict_of_objectives=dict_of_objectives
        )

    def run_experiment_step_submitted(self) -> None:

        # Note for the future: Could consider doing two Optimiser and two Experiment classes instead of these checks
        assert issubclass(type(self.batch_manager), SubmitBatchManager), (
            "Batch manager must be subclassing SubmitBatchManager to call this method"
        )

        if not self.current_step == 0:

            self.batch_manager.wait_for_jobs(  # type: ignore[union-attr]  # Checked above
                experimental_state=self.state
            )

            results = self.state.get_results(
                start_point=self.current_batch_indices['start'],
                end_point=self.current_batch_indices['end']
            )

            dict_of_objectives = self.result_processor.process(
                results=results
            )
            dict_of_parameters = self.state.get_parameters(
                start_point=self.current_batch_indices['start'],
                end_point=self.current_batch_indices['end']
            )

            self.save_objectives_to_state(dict_of_objectives=dict_of_objectives)
            self.send_objectives_to_optimiser(
                dict_of_parameters=dict_of_parameters,
                dict_of_objectives=dict_of_objectives
            )

        self.optimiser.run_optimisation_step()

        dict_of_parameters = self.get_parameters_from_optimiser()

        self._save_optimiser()

        if not self.current_step == (self.n_total_steps - 1):
            self.batch_manager.submit_batch(  # type: ignore[union-attr]  # Checked above
                dict_of_parameters=dict_of_parameters,
                experimental_state=self.state
            )

    def run_experiment_step(self) -> None:
        if self.experiment_config.experiment_mode == ExperimentMode.local:
            self.run_experiment_step_direct()

        elif self.experiment_config.experiment_mode == ExperimentMode.local_slurm:
            self.run_experiment_step_submitted()

        elif self.experiment_config.experiment_mode == ExperimentMode.remote_slurm:
            self.run_experiment_step_submitted()

        else:
            raise RuntimeError(f"Unknown experiment mode: '{self.experiment_config.experiment_mode}'")

    def run_experiment(self) -> None:

        n_remaining_steps = self.n_total_steps - self.current_step

        for i in range(n_remaining_steps):
            self.run_experiment_step()

    @property
    def current_step(self) -> int:
        return self.n_points_submitted // self.n_evaluations_per_step

    @property
    def n_total_steps(self) -> int:
        total_full_steps = (self.n_initial_points + self.n_bayesian_points) // self.n_evaluations_per_step
        return total_full_steps + 1

    @property
    def n_evaluations_per_step(self) -> int:
        return self.optimiser.n_evaluations_per_step

    @property
    def n_initial_points(self) -> int:
        return self.optimiser.n_initial_points

    @property
    def n_bayesian_points(self) -> int:
        return self.optimiser.n_bayesian_points

    @property
    def n_points_submitted(self) -> int:
        return self.state.n_points

    @property
    def n_points_evaluated(self) -> int:
        return self.optimiser.n_points_evaluated

    @property
    def current_batch_indices(self) -> dict[str, int]:
        return {
            'start': self.n_points_evaluated,
            'end': self.n_points_evaluated + self.n_evaluations_per_step - 1
        }

    def restart_experiment(self) -> None:
        raise NotImplementedError("Restarting an experiment is not implemented yet.")
