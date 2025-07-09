from veropt.interfaces.experiment import Experiment
from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig
from veropt.interfaces.result_processing import TestVerosResultProcessor
from veropt.interfaces.experiment_utility import ExperimentConfig

simulation_config = LocalVerosConfig.load("tests/interfaces/configs/local_veros_config.json")
simulation_runner = LocalVerosRunner(config=simulation_config)

optimiser_config = "tests/interfaces/configs/optimiser_config.json"
experiment_config = ExperimentConfig.load("tests/interfaces/configs/veros_experiment_config.json")

result_processor = TestVerosResultProcessor(objective_names=experiment_config.objective_names)

experiment = Experiment(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

experiment.run_experiment()
