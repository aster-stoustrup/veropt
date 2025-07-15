import tempfile

from veropt.interfaces.experiment import Experiment
from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.result_processing import MockResultProcessor
from veropt.interfaces.experiment_utility import OptimiserConfig, ExperimentConfig


def test_experiment() -> None:

    optimiser_config = OptimiserConfig(
        n_initial_points=1,
        n_bayesian_points=1,
        n_evaluations_per_step=1
    )

    experiment_config = ExperimentConfig.load("veropt/interfaces/configs/experiment_config.json")

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_script_root_directory = tmp_dir
        run_script_filename = "foo"
        run_script = f"{run_script_root_directory}/{run_script_filename}.txt"

        with open(run_script, "w+") as f:
            f.write("bar")

        experiment_config.path_to_experiment = tmp_dir
        experiment_config.run_script_root_directory = tmp_dir
        experiment_config.run_script_filename = run_script_filename

        objective_names = ["objective1"]
        objectives = {"objective1": 1.}

        simulation_config = MockSimulationConfig()
        simulation_config.output_filename = "output"
        simulation_config.output_directory = ""
        simulation_runner = MockSimulationRunner(config=simulation_config)

        result_processor = MockResultProcessor(
            objective_names=objective_names,
            objectives=objectives)

        experiment = Experiment(
            simulation_runner=simulation_runner,
            result_processor=result_processor,
            experiment_config=experiment_config,
            optimiser_config=optimiser_config
        )

        experiment.run_experiment()


test_experiment()
