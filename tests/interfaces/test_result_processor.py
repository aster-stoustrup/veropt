from veropt.interfaces.result_processing import MockResultProcessor
from veropt.interfaces.simulation import SimulationResult
import tempfile
import numpy as np


def test_process() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:

        ids = ["point=0", "point=1", "point=2", "point=3"]
        output_filenames = ["output_0", "error_output", "output_2", "output_3"]
        return_codes = [0, 0, 1, 0]

        parameters = {"param1": 1.0}
        stdout_file = "stdout.txt"
        stderr_file = "stderr.txt"

        for name in output_filenames[1:]:
            with open(f"{tmp_dir}/{name}.txt", "w") as f:
                f.write("0.1")

        simulation_results_dict = {
            i: SimulationResult(
                simulation_id=ids[i],
                parameters=parameters,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                output_directory=tmp_dir,
                output_filename=output_filenames[i],
                return_code=return_codes[i]
            ) for i in range(len(ids))
        }

        objective_name = "objective1"
        objective_names = [objective_name]
        objectives = {"objective1": 0.0}

        processor = MockResultProcessor(
            objectives=objectives,
            objective_names=objective_names,
            fixed_objective=True)

        objectives_dict = processor.process(results=simulation_results_dict)

        for i in range(3):
            assert np.isnan(objectives_dict[i][objective_name])
        assert objectives_dict[3] == objectives
