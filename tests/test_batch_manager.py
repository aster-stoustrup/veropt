from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.batch_manager import LocalBatchManager, LocalBatchManagerConfig


simulation_config = MockSimulationConfig(
    cfg1="1",
    cfg2="2"
)

batch_config = LocalBatchManagerConfig(
    run_script_filename="test_experiment",
    run_script_root_directory="/Users/martamrozowska/Desktop/veropt_testing/exp_test_experiment/test_experiment_setup",
    experiment_directory="/Users/martamrozowska/Desktop/veropt_testing/exp_test_experiment",
    n_evals_per_step=1,
    next_point=0,
    max_workers=3
)

parameters = {0 : {"c_k": 0.05, "c_eps": 1.0},
              1 : {"c_k": 0.1,  "c_eps": 0.5}, 
              2 : {"c_k": 0.2,  "c_eps": 0.1}}

runner = MockSimulationRunner(config=simulation_config)
batch_manager = LocalBatchManager(
    config=batch_config,
    simulation_runner=runner,
)

results = batch_manager.run_batch(dict_of_parameters=parameters)
for result in results.values():
    print("Output dict:", result.model_dump())
    print("Return code:", result.return_code)
    print("Stdout file:", result.stdout_file)
    print("Stderr file:", result.stderr_file)
    