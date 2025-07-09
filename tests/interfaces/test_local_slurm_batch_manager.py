from veropt.interfaces.slurm_simulation import SlurmVerosConfig, SlurmVerosRunner
from veropt.interfaces.batch_manager import LocalSlurmBatchManager
from veropt.interfaces.experiment_utility import ExperimentalState, Point

runner_config = SlurmVerosConfig.load("tests/configs/slurm_veros_config.json")
simulation_runner = SlurmVerosRunner(config=runner_config)

batch_manager = LocalSlurmBatchManager(
    simulation_runner=simulation_runner,
    run_script_filename="acc",
    run_script_root_directory="/groups/ocean/mmroz/veropt_dev/test_sbatch/source",
    results_directory="/groups/ocean/mmroz/veropt_dev/test_sbatch/results",
    output_filename="test",
    check_job_status_sleep_time=10
)

dict_of_parameters = {
    0: {"c_k": 0.8},
    1: {"c_k": 0.3}
}

points = {i: Point(
    parameters=parameters,
    state="Received parameters from core"
) for i, parameters in dict_of_parameters.items()
}

experimental_state = ExperimentalState.make_fresh_state(
    experiment_name="test_experiment",
    experiment_directory="/groups/ocean/mmroz/veropt_dev/test_sbatch",
    state_json="/groups/ocean/mmroz/veropt_dev/test_sbatch/results/test_state.json",
    points=points,
    next_point=2
)

batch_manager.run_batch(
    dict_of_parameters=dict_of_parameters,
    experimental_state=experimental_state
)