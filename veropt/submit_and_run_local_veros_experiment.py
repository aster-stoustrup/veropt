from veropt.interfaces.batch_manager import LocalBatchManager, LocalBatchManagerConfig
from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig


sim_cfg = LocalVerosConfig(
    env_manager= "conda",
    env_name= "veros",
    path_to_env= "/Users/martamrozowska/miniforge3",
    veros_path= "veros",
    backend="numpy",
    device= "cpu",
    float_type= "float64",
    keep_old_params=False,
)

batch_config = LocalBatchManagerConfig(
    run_script_filename="test_experiment",
    run_script_root_directory="/Users/martamrozowska/Desktop/veropt_testing/exp_test_experiment/test_experiment_setup",
    experiment_directory="/Users/martamrozowska/Desktop/veropt_testing/exp_test_experiment",
    n_evals_per_step=1,
    next_point=0,
    max_workers=3
)

parameters = {0: {"c_k": 0.05, "c_eps": 1.0},
              1: {"c_k": 0.1,  "c_eps": 0.5}, 
              2: {"c_k": 0.2,  "c_eps": 0.1}}

runner = LocalVerosRunner(config=sim_cfg)
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
