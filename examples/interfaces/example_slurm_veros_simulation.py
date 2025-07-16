from veropt.interfaces.slurm_simulation import SlurmVerosConfig, SlurmVerosRunner

config = SlurmVerosConfig.load("veropt/interfaces/configs/slurm_veros_config.json")

simulation_runner = SlurmVerosRunner(config=config)

result = simulation_runner.save_set_up_and_run(
    simulation_id="test_simulation",
    parameters={"c_k": 0.2},
    run_script_directory="path/to/run/scrip",
    run_script_filename="acc",
    output_filename="test"
)

print(result.model_dump())
