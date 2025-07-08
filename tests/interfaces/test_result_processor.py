from veropt.interfaces.result_processing import MockResultProcessor
from veropt.interfaces.simulation import SimulationResult
import tempfile
import os
import math


with tempfile.TemporaryDirectory() as tmp_dir:

    ids = ["point=0", "point=1", "point=2", "point=3"]
    output_filenames = ["output_0.nc", "error_output.nc", "output_2.nc", "output_3.nc"]
    return_codes = [0, 0, 1, 0]

    parameters = {i: {"param1": 1.0} for i in range(4)}
    stdout_files = ["stdout.txt" for i in range(4)]
    stderr_files = ["stderr.txt" for i in range(4)]
    output_files = [os.path.join(tmp_dir, f) for f in output_filenames]

    for i in range(1, 4):
        with open(output_files[i], "w") as f:
            f.write("dummy content")

    simulation_results_dict = {
        i: SimulationResult(
            simulation_id=ids[i],
            parameters=parameters[i],
            stdout_file=stdout_files[i],
            stderr_file=stderr_files[i],
            output_file=output_files[i],
            return_code=return_codes[i]
        ) for i in range(len(ids))
    }

    objective_names = ["objective1", "objective2"]
    objectives = {name: 0.0 for name in objective_names}
    nan_objectives = {name: float("nan") for name in objective_names}

    processor = MockResultProcessor(
        objectives=objectives,
        objective_names=objective_names)

    objectives_dict = processor.process(results=simulation_results_dict)

    assert [objectives_dict[i] == nan_objectives for i in range(3)]
    assert objectives_dict[3] == objectives

    simulation_results_dict_with_lists = {
        0: [simulation_results_dict[0], simulation_results_dict[1]],
        1: [simulation_results_dict[2], simulation_results_dict[3]],
        2: [simulation_results_dict[3], simulation_results_dict[3]],
    }

    objectives_dict = processor.process(results=simulation_results_dict_with_lists)

    assert [objectives_dict[i] == nan_objectives for i in range(2)]
    assert objectives_dict[2] == objectives
