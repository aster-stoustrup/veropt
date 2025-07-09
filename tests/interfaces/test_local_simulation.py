from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig
import xarray as xr

# conda_env = Conda(
#     path_to_env="/Users/martamrozowska/miniforge3",
#     env_name="veros",
#     command="echo $CONDA_DEFAULT_ENV"
# )
# simulation = VerosSimulation(env_manager=conda_env)
# result = simulation.run()
# print("Output:", result['simulation_output'])
# print("Error:", result['simulation_error'])

parameters = {
    "c_k": 0.05,
    "c_eps": 1.0
}
local_config = LocalVerosConfig.load('tests/interfaces/configs/local_veros_config.json')

simulation_runner = LocalVerosRunner(config=local_config)
result = simulation_runner.save_set_up_and_run(
    simulation_id="point=0",
    parameters=parameters,
    run_script_directory="/Users/martamrozowska/Desktop/veros/acc",
    run_script_filename="acc",
    output_filename="acc")
print("Output dict:", result.model_dump())

# ds = xr.open_dataset(result.output_filename)
# print(ds)
