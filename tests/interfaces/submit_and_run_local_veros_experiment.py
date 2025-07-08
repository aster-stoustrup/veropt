from veropt.interfaces.batch_manager import LocalBatchManager, LocalBatchManagerConfig
from veropt.interfaces.local_simulation import LocalVerosRunner, LocalVerosConfig
from veropt.interfaces.experiment_utility import ExperimentalState, Point


parameters = {0: {"c_k": 0.05, "c_eps": 1.0},
              1: {"c_k": 0.1,  "c_eps": 0.5},
              2: {"c_k": 0.2,  "c_eps": 0.1}}

points = {i: Point(
    parameters=val,
    state="Received points from core"
) for i, val in parameters.items()}

simulation_config = LocalVerosConfig.load('configs/local_veros_config.json')
batch_config = LocalBatchManagerConfig.load('configs/local_batch_manager_config.json')
experimental_state = ExperimentalState.make_fresh_state(
    experiment_name="test_experiment",
    experiment_directory="/Users/martamrozowska/Desktop/veropt_testing/exp_test_experiment",
    state_json="/Users/martamrozowska/Desktop/veropt_testing/exp_test_experiment/experimental_state.json",
    points=points,
    next_point=3,
)

runner = LocalVerosRunner(config=simulation_config)
batch_manager = LocalBatchManager(
    config=batch_config,
    simulation_runner=runner,
)

results = batch_manager.run_batch(
    dict_of_parameters=parameters,
    experimental_state=experimental_state
)

for result in results.values():
    print("Output dict:", result.model_dump())
    print("Return code:", result.return_code)
    print("Stdout file:", result.stdout_file)
    print("Stderr file:", result.stderr_file)
