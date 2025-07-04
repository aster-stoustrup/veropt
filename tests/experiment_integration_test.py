from veropt.interfaces.experiment import Experiment
from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.result_processing import MockResultProcessor

objective_names = ["objective1"]
objectives = {"objective1": 0.0}

simulation_config = MockSimulationConfig.load("/Users/martamrozowska/Desktop/veropt/tests/configs/mock_simulation_config.json")
simulation_runner = MockSimulationRunner(config=simulation_config)

result_processor = MockResultProcessor(
    objectives=objectives,
    objective_names=objective_names
)

optimiser_config = "/Users/martamrozowska/Desktop/veropt/tests/configs/optimiser_config.json"
experiment_config = "/Users/martamrozowska/Desktop/veropt/tests/configs/experiment_config.json"

experiment = Experiment(
    simulation_runner=simulation_runner,
    result_processor=result_processor,
    experiment_config=experiment_config,
    optimiser_config=optimiser_config
)

experiment.run_experiment()