from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig
from pydantic import BaseModel 

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
local_config = LocalVerosConfig(
    env_manager= "conda",
    env_name= "veros",
    path_to_env= "/Users/martamrozowska/miniforge3",
    veros_path= "veros",
    backend="numpy",
    device= "cpu",
    float_type= "float64",
    keep_old_params=True,
)

simulation_runner = LocalVerosRunner(
    cfg=local_config
)
result = simulation_runner.set_up_and_run(
    id="point=0", 
    parameters=parameters,
    setup_path="/Users/martamrozowska/Desktop/veros/acc",
    setup_name="acc")
print("Output dict:", result.model_dump())