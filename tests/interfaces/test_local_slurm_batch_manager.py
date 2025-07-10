import os
from veropt.interfaces.slurm_simulation import SlurmVerosConfig, SlurmVerosRunner
from veropt.interfaces.batch_manager import LocalSlurmBatchManager
from veropt.interfaces.experiment_utility import ExperimentalState


def test_local_slurm_batch_manager():

    if "SLURM_JOB_ID" in os.environ:

        runner_config = SlurmVerosConfig.load("interfaces/configs/slurm_veros_config.json")
        simulation_runner = SlurmVerosRunner(config=runner_config)

        batch_manager = LocalSlurmBatchManager(
            simulation_runner=simulation_runner,
            run_script_filename="acc",
            run_script_root_directory="/path/to/run/script/root",
            results_directory="/path/to/results",
            output_filename="test",
            check_job_status_sleep_time=10
        )

        dict_of_parameters = {
            0: {"c_k": 0.8},
            1: {"c_k": 0.3}
        }

        experimental_state = ExperimentalState.make_fresh_state(
            experiment_name="test_experiment",
            experiment_directory="/groups/ocean/mmroz/veropt_dev/test_sbatch",
            state_json="/groups/ocean/mmroz/veropt_dev/test_sbatch/results/test_state.json"
        )

        batch_manager.run_batch(
            dict_of_parameters=dict_of_parameters,
            experimental_state=experimental_state
        )
