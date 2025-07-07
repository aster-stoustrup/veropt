from veropt.interfaces.slurm_simulation import SlurmVerosConfig, SlurmVerosRunner
from veropt.interfaces.batch_manager import LocalSlurmBatchManager, LocalSlurmBatchManagerConfig
from veropt.interfaces.experiment_utility import ExperimentalState, Point

runner_config = SlurmVerosConfig.load("/groups/ocean/mmroz/veropt_dev/veropt/tests/configs/slurm_veros_config.json")
simulation_runner = SlurmVerosRunner(config=runner_config)

batch_manager_config = LocalSlurmBatchManagerConfig.load("/groups/ocean/mmroz/veropt_dev/veropt/tests/configs/local_slurm_batch_manager_config.json")
batch_manager = LocalSlurmBatchManager(
    simulation_runner=simulation_runner,
    config=batch_manager_config
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