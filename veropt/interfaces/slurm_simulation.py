import os
import stat
import time
import subprocess
from typing import Literal

from veropt.interfaces.simulation import Simulation, SimulationResult, SimulationRunner
from veropt.interfaces.utility import Config
from veropt.interfaces.veros_utility import edit_veros_run_script


def write_batch_script_string(
        batch_script_template: str,
        template_substitutions: dict
) -> str:

    assert os.path.isfile(batch_script_template), "Batch script template not found."

    with open(batch_script_template, "r") as f:
        batch_script_string = f.read()

    return batch_script_string.format(**template_substitutions)


def create_batch_script(
        batch_script_file: str,
        batch_script_string: str
) -> None:

    with open(batch_script_file, 'w+') as f:
        f.write(batch_script_string)

    st = os.stat(batch_script_file)
    os.chmod(batch_script_file, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def try_to_run(
        simulation: Simulation,
        parameters: dict[str, float],
        max_tries: int
) -> SimulationResult:

    success = False
    tries = 0

    while not success and tries < max_tries:
        result = simulation.run(parameters=parameters)

        if os.path.getsize(result.stderr_file) == 0:
            print(f"Simulation {result.simulation_id} submitted successfully.")
            success = True

        else:
            with open(result.stderr_file, "r") as f:
                error = f.read()

            print(f"Submission of simulation {result.simulation_id} failed.")
            print(f"Error output was:\n{error}")

            tries += 1

            print(f"Retrying in {tries * 60} seconds.")
            time.sleep(tries * 60)

    return result


class SlurmSimulation(Simulation):
    def __init__(
            self,
            simulation_id: str,
            run_script_directory: str,
            output_filename: str,
            batch_script_file: str
    ):

        self.id = simulation_id
        self.run_script_directory = run_script_directory
        self.output_filename = output_filename
        self.batch_script_file = batch_script_file
        command = f"sbatch --parsable {batch_script_file}"
        self.command_arguments = command.split(" ")

    def run(
            self,
            parameters: dict[str, float]
    ) -> SimulationResult:

        assert os.path.isfile(self.batch_script_file), "Batch script not found."

        process = subprocess.run(
            args=self.command_arguments,
            cwd=self.run_script_directory,
            capture_output=True,
            text=True
        )

        stdout_file = os.path.join(self.run_script_directory, f"{self.id}.out")
        stderr_file = os.path.join(self.run_script_directory, f"{self.id}.err")

        with open(stdout_file, "w") as f:
            f.write(process.stdout)

        with open(stderr_file, "w") as f:
            f.write(process.stderr)

        return SimulationResult(
            simulation_id=self.id,
            parameters=parameters,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            output_directory=self.run_script_directory,
            output_filename=self.output_filename
        )


class SlurmVerosConfig(Config):
    batch_script_template: str
    partition_name: str
    group_name: str
    constraint: str
    n_cores: int
    n_cycles: int
    cycle_length: int
    backend: Literal["jax", "numpy"]
    n_cores_nx: int
    n_cores_ny: int
    float_type: Literal["float32", "float64"]
    max_tries: int
    keep_old_params: bool = False


class SlurmVerosRunner(SimulationRunner):
    def __init__(
            self,
            config: SlurmVerosConfig
    ):
        self.config = config

    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: str,
            output_filename: str
    ) -> SimulationResult:

        run_script_file = os.path.join(run_script_directory, f"{run_script_filename}.py")
        batch_script_filename = f"veros_batch_{simulation_id}"
        batch_script_file = os.path.join(run_script_directory, f"{batch_script_filename}.sh")
        slurm_log_filename = f"slurm_{simulation_id}"
        slurm_log_file = os.path.join(run_script_directory, f"{slurm_log_filename}.out")

        edit_veros_run_script(
            run_script=run_script_file,
            parameters=parameters
        ) if not self.config.keep_old_params else None

        substitutions_dict = {
            "simulation_id": simulation_id,
            "run_script_filename": run_script_filename,
            "batch_script_filename": batch_script_filename,
            "slurm_log_filename": slurm_log_filename,
            "output_filename": output_filename
        }

        template_substitutions = self.config.model_dump() | substitutions_dict

        batch_script_string = write_batch_script_string(
            batch_script_template=self.config.batch_script_template,
            template_substitutions=template_substitutions
        )

        create_batch_script(
            batch_script_file=batch_script_file,
            batch_script_string=batch_script_string
        )

        simulation = SlurmSimulation(
            simulation_id=simulation_id,
            run_script_directory=run_script_directory,
            output_filename=output_filename,
            batch_script_file=batch_script_file
        )

        result = try_to_run(
            simulation=simulation,
            parameters=parameters,
            max_tries=self.config.max_tries
        )

        result.slurm_log_file = slurm_log_file

        return result
