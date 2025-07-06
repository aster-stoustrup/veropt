from abc import ABC, abstractmethod
from enum import Enum
import json
import os
import subprocess
from typing import Optional, Unpack, Union, TypedDict, Literal, Dict, List
from veropt.interfaces.simulation import SimulationRunnerConfig, SimulationResult, Simulation, SimulationRunner
from veropt.interfaces.veros_utility import edit_veros_run_script
from veropt.interfaces.utility import Config

import torch
from pydantic import BaseModel


class EnvManager(ABC):
    def __init__(
            self,
            path_to_env: str,
            env_name: str,
            command: str
    ) -> None:

        self.path_to_env = path_to_env
        self.env_name = env_name
        self.command = command

    @abstractmethod
    def run_in_env(self) -> subprocess.CompletedProcess:
        ...


class Conda(EnvManager):
    def run_in_env(self) -> subprocess.CompletedProcess:

        # "path_to_env" is the path to the conda installation, not the environment
        full_command = f"source {self.path_to_env}/bin/activate {self.env_name} && {self.command}"

        return subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )


class Venv(EnvManager):
    def run_in_env(self) -> subprocess.CompletedProcess:

        full_command = f"source {self.path_to_env}/bin/activate && {self.command}"

        return subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )


class LocalSimulation(Simulation):
    """Run a simulation in a specified environment as a subprocess."""
    def __init__(
            self,
            simulation_id: str,
            run_script_directory: str,
            env_manager: EnvManager,
            output_file: str
    ) -> None:

        self.id = simulation_id
        self.run_script_directory = run_script_directory
        self.env_manager = env_manager
        self.output_file = output_file

    # TODO: Should there be an option to supress output files?
    def run(
            self,
            parameters: dict[str, float]
    ) -> SimulationResult:

        result = self.env_manager.run_in_env()

        stdout_file = os.path.join(self.run_script_directory, f"{self.id}.out")
        stderr_file = os.path.join(self.run_script_directory, f"{self.id}.err")

        with open(stdout_file, "w") as f_out:
            f_out.write(result.stdout)

        with open(stderr_file, "w") as f_err:
            f_err.write(result.stderr)

        return SimulationResult(
            simulation_id=self.id,
            parameters=parameters,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            return_code=result.returncode,
            output_directory=self.run_script_directory,
            output_filename=self.output_file
        )


class MockSimulationConfig(Config):
    stdout_file: list[str] = ["test_stdout.txt"]
    stderr_file: list[str] = ["test_stderr.txt"]
    return_code: list[int] = [0]
    output_filename: list[str] = ["test_output.nc"]
    return_list: bool = False


class MockSimulationRunner(SimulationRunner):
    """A mock simulation runner for testing purposes."""
    def __init__(
            self,
            config: MockSimulationConfig
    ) -> None:
        self.config = config

    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str = "",
            run_script_filename: str = "",
            output_filename: str = ""
    ) -> Union[SimulationResult, list[SimulationResult]]:

        print(f"Running test simulation with parameters: {parameters} and config: {self.config.model_dump()}")

        if self.config.return_list:

            stdout_files = self.config.stdout_file
            stderr_files = self.config.stderr_file
            return_codes = self.config.return_code
            output_filenames = self.config.output_filename

            zipped_lists = zip(stdout_files, stderr_files, return_codes, output_filenames)

            results = [SimulationResult(
                simulation_id=simulation_id,
                parameters=parameters,
                stdout_file=out_file,
                stderr_file=err_file,
                output_directory="",
                output_filename=output_filename,
                return_code=return_code
            ) for out_file, err_file, return_code, output_filenames in zipped_lists]

            return results

        else:
            return SimulationResult(
                simulation_id=simulation_id,
                parameters=parameters,
                stdout_file=self.config.stdout_file[0],
                stderr_file=self.config.stderr_file[0],
                output_directory="",
                output_filename=self.config.output_filename[0],
                return_code=self.config.return_code[0]
            )


class LocalVerosConfig(Config):
    env_manager: Literal["conda", "venv"]
    env_name: str
    path_to_env: str
    veros_path: str
    backend: Literal["jax", "numpy"]
    device: Literal["cpu", "gpu"]
    float_type: Literal["float32", "float64"]
    command: Optional[str] = None
    keep_old_params: bool = False


class LocalVerosRunner(SimulationRunner):
    """Set up and run a Veros simulation in a local environment."""
    def __init__(
            self,
            config: LocalVerosConfig
    ) -> None:
        self.config = config

    # TODO: Should there be a way to override the command to include custom settings?
    def _make_command(
            self,
            run_script: str,
            run_script_directory: str
    ) -> str:
        gpu_string = f"--backend {self.config.backend} --device {self.config.device}"
        # TODO: Using "veros_path" here is misleading. It should be the path to the Veros executable.
        command = f"cd {run_script_directory} && {self.config.veros_path} run {gpu_string}" \
                  f" --float-type {self.config.float_type} {run_script}"
        return command

    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: dict[str, float],
            run_script_directory: str,
            run_script_filename: str,
            output_filename: str
    ) -> SimulationResult:
        
        run_script = os.path.join(run_script_directory, f"{run_script_filename}.py")

        edit_veros_run_script(
            run_script=run_script, 
            parameters=parameters
        ) if not self.config.keep_old_params else None

        command = self._make_command(
            run_script=run_script, 
            run_script_directory=run_script_directory
        ) if self.config.command is None else self.config.command

        # TODO: This is bad. It should be a factory method or similar?
        env_manager_classes = {
            "conda": Conda,
            "venv": Venv,
        }
        EnvManagerClass = env_manager_classes[self.config.env_manager]

        env_manager = EnvManagerClass(
            path_to_env=self.config.path_to_env,
            env_name=self.config.env_name,
            command=command
        )

        simulation = LocalSimulation(
            simulation_id=simulation_id,
            run_script_directory=run_script_directory,
            env_manager=env_manager,
            output_filename=output_filename
        )

        result = simulation.run(parameters=parameters)

        return result
