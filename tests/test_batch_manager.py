from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.batch_manager import batch_manager, ExperimentMode
from veropt.interfaces.experiment_utility import ExperimentalState


experiment_name = "test_experiment"
experiment_directory = f"/Users/martamrozowska/Desktop/veropt_testing/exp_{experiment_name}"
results_directory = f"{experiment_directory}/results"
state_json = f"{experiment_directory}/experimental_state.json"
run_script_root_directory = f"{experiment_directory}/{experiment_name}_setup"


parameters = {0: {"c_k": 0.05, "c_eps": 1.0},
              1: {"c_k": 0.1,  "c_eps": 0.5},
              2: {"c_k": 0.2,  "c_eps": 0.1}
              }

simulation_config = MockSimulationConfig.load("/Users/martamrozowska/Desktop/veropt/tests/configs/mock_simulation_config.json")
experimental_state = ExperimentalState.make_fresh_state(
    experiment_name=experiment_name,
    experiment_directory=experiment_directory,
    state_json=state_json
    )

runner = MockSimulationRunner(config=simulation_config)
my_batch_manager = batch_manager(
    experiment_mode=ExperimentMode.LOCAL,
    simulation_runner=runner,
    run_script_filename=experiment_name,
    run_script_root_directory=run_script_root_directory,
    results_directory=results_directory,
    output_filename=""
    )

results = my_batch_manager.run_batch(
    dict_of_parameters=parameters,
    experimental_state=experimental_state
    )

for result in results.values():
    print("Output dict:", [r.model_dump() for r in result])
