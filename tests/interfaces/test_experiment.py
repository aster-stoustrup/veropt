import tempfile

from veropt.optimiser.saver_loader_utility import rehydrate_object
from veropt.optimiser.objective import Objective
from veropt.interfaces.experiment import ExperimentObjective, Experiment, _mask_nans
from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.result_processing import MockResultProcessor
from veropt.interfaces.experiment_utility import OptimiserConfig, ExperimentConfig, ExperimentalState, Point

import numpy as np


def test_experiment_objective() -> None:
    bounds_lower, bounds_upper = [0.1], [10.0]
    n_variables, n_objectives = 1, 1
    variable_names, objective_names = ["var1"], ["obj1"]
    suggested_parameters_json = "suggested.json"
    evaluated_objectives_json = "evaluated.json"

    experiment_objective = ExperimentObjective(
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
        n_variables=n_variables,
        n_objectives=n_objectives,
        objective_names=objective_names,
        variable_names=variable_names,
        suggested_parameters_json=suggested_parameters_json,
        evaluated_objectives_json=evaluated_objectives_json
    )

    saved_state = experiment_objective.gather_dicts_to_save()
    rehydrated_experiment_objective = rehydrate_object(
        superclass=Objective,
        name=saved_state["name"],
        saved_state=saved_state
    )

    assert isinstance(rehydrated_experiment_objective, ExperimentObjective)


def test_mask_nans() -> None:

    experimental_state = ExperimentalState.make_fresh_state(
        experiment_name="",
        experiment_directory="",
        state_json=""
    )

    values = [20.0, 0.5, 17.0, -0.01, -13.0]
    initial_dict_of_objectives = {i: {"obj1": value} for i, value in enumerate(values)}

    for objective_values in initial_dict_of_objectives.values():
        point = Point(
            parameters={"param1": 0.1},
            state="",
            objective_values=objective_values
        )

        experimental_state.update(point)

    new_values = [5.0, float("nan"), -5.0]
    dict_of_objectives = {i: {"obj1": value} for i, value in enumerate(new_values)}

    for objective_values in dict_of_objectives.values():
        point = Point(
            parameters={"param1": 0.1},
            state="",
            objective_values=objective_values
        )

        experimental_state.update(point)

    updated_dict_of_objectives = _mask_nans(
        dict_of_objectives=dict_of_objectives,
        experimental_state=experimental_state
    )

    assert not np.isnan(updated_dict_of_objectives[1]["obj1"])

    new_experimental_state = ExperimentalState.make_fresh_state(
        experiment_name="",
        experiment_directory="",
        state_json=""
    )

    try:
        _mask_nans(
            dict_of_objectives=dict_of_objectives,
            experimental_state=new_experimental_state
        )
    except AssertionError:
        assert True
    else:
        assert False


def test_experiment_step() -> None:

    optimiser_config = OptimiserConfig(
        n_initial_points=1,
        n_bayesian_points=1,
        n_evaluations_per_step=1
    )

    experiment_config = ExperimentConfig(
        experiment_name="integration_test",
        parameter_names=["param1"],
        parameter_bounds={"param1": [0, 1]},
        path_to_experiment="path/to/experiment",
        experiment_mode="local",
        run_script_filename="test_experiment",
        output_filename="output"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_script_root_directory = tmp_dir
        run_script_filename = "foo"
        run_script = f"{run_script_root_directory}/{run_script_filename}.txt"

        with open(run_script, "w+") as f:
            f.write("bar")

        experiment_config.path_to_experiment = tmp_dir
        experiment_config.run_script_root_directory = tmp_dir
        experiment_config.run_script_filename = run_script_filename

        objective_names = ["obj1"]
        objectives = {"obj1": 1.}

        simulation_config = MockSimulationConfig()
        simulation_config.output_filename = "output"
        simulation_config.output_directory = tmp_dir
        simulation_runner = MockSimulationRunner(config=simulation_config)

        with open(f"{tmp_dir}/output.txt", "w+") as f:
            f.write("0.1")

        result_processor = MockResultProcessor(
            objective_names=objective_names,
            objectives=objectives)

        experiment = Experiment(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=optimiser_config
        )

        experiment.run_experiment_step()

        assert experiment.state.points[0].objective_values["obj1"] == 0.1  # type: ignore


def test_experiment() -> None:

    # TODO: make a test for integrated experiment
    #       need better MockSimulation and MockResultProcessor for this

    pass
