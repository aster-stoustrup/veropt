from veropt.interfaces.constructors import experiment
from veropt.interfaces.simulation import SimulationRunner


class UserDefinedSimulationRunner(SimulationRunner):
    pass


simulation_config = LocalVerosConfig.load("veropt/interfaces/configs/local_veros_config.json")
simulation_runner = LocalVerosRunner(config=simulation_config)

optimiser_config = "veropt/interfaces/configs/optimiser_config.json"
experiment_config = "veropt/interfaces/configs/veros_experiment_config.json"

result_processor = TestVerosResultProcessor(objective_names=["amoc"])

veros_experiment = experiment(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

veros_experiment.run_experiment()