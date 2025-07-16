from veropt.interfaces.experiment import Experiment
from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig
from veropt.interfaces.result_processing import TestVerosResultProcessor

simulation_config = LocalVerosConfig.load("veropt/interfaces/configs/local_veros_config.json")
simulation_runner = LocalVerosRunner(config=simulation_config)

optimiser_config = "veropt/interfaces/configs/optimiser_config.json"
experiment_config = "veropt/interfaces/configs/veros_experiment_config.json"

result_processor = TestVerosResultProcessor(objective_names=["amoc"])

experiment = Experiment(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

experiment.run_experiment()
