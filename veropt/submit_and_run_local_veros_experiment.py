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

batch_cfg = LocalBatchManagerConfig(
    experiment_id="test_experiment",
    path_to_experiment="/Users/martamrozowska/Desktop/veropt_testing",
    latest_point=0,
    max_workers=3
)

parameters = [{"c_k": 0.05, "c_eps": 1.0},
              {"c_k": 0.1,  "c_eps": 0.5}, 
              {"c_k": 0.2,  "c_eps": 0.1}]

runner = LocalVerosRunner(cfg=sim_cfg)
batch_manager = LocalBatchManager(
    cfg=batch_cfg,
    simulation_runner=runner,
    list_of_parameters=parameters,
)

results = batch_manager.run_batch()
for result in results:
    print("Output dict:", result.model_dump())
    print("Return code:", result.return_code)
    print("Stdout file:", result.stdout_file)
    print("Stderr file:", result.stderr_file)