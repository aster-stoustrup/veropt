import tempfile
import os

from veropt.interfaces.local_simulation import MockSimulationRunner, MockSimulationConfig
from veropt.interfaces.batch_manager import batch_manager, ExperimentMode
from veropt.interfaces.experiment_utility import ExperimentalState


def test_local_batch_manager() -> None:

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment_name = "test_experiment"
        experiment_directory = f"{tmp_dir}/{experiment_name}"
        results_directory = f"{experiment_directory}/results"
        state_json = f"{experiment_directory}/experimental_state.json"
        run_script_root_directory = tmp_dir
        run_script_filename = "foo"
        run_script = f"{run_script_root_directory}/{run_script_filename}.txt"

        with open(run_script, "w+") as f:
            f.write("bar")

        parameters = {
            0: {"c_k": 0.05, "c_eps": 1.0},
            1: {"c_k": 0.1,  "c_eps": 0.5},
            2: {"c_k": 0.2,  "c_eps": 0.1}
        }

        experimental_state = ExperimentalState.make_fresh_state(
            experiment_name=experiment_name,
            experiment_directory=experiment_directory,
            state_json=state_json
        )

        simulation_config = MockSimulationConfig()
        runner = MockSimulationRunner(config=simulation_config)

        my_batch_manager = batch_manager(
            experiment_mode=ExperimentMode.LOCAL,
            simulation_runner=runner,
            run_script_filename=run_script_filename,
            run_script_root_directory=run_script_root_directory,
            results_directory=results_directory,
            output_filename=""
        )

        results = my_batch_manager.run_batch(
            dict_of_parameters=parameters,
            experimental_state=experimental_state
        )

        assert [os.path.isdir(f"{results_directory}/point={i}") for i in parameters.keys()]
        assert [os.path.isfile(f"{results_directory}/point={i}/foo.txt") for i in parameters.keys()]

        assert [experimental_state.points[i].result == results[i] for i in parameters.keys()]
        assert os.path.isfile(state_json)

        loaded_experimental_state = ExperimentalState.load(state_json)

        assert loaded_experimental_state.model_dump() == experimental_state.model_dump()
