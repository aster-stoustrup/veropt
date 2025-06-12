from abc import ABC, abstractmethod
from enum import Enum
import json
import os
import subprocess
from typing import Optional, Unpack, Union, TypedDict, Literal, Dict, List
from pydantic import BaseModel, Field
from veropt.interfaces.simulation import SimulationRunnerConfig, SimulationResult, Simulation, SimulationRunner


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
    def run_in_env(
            self
    ) -> subprocess.CompletedProcess:
        ...


class Conda(EnvManager):
    def run_in_env(
            self
    ) -> subprocess.CompletedProcess:
        
        # "path_to_env" is the path to the conda installation, not the environment
        full_command = f"source {self.path_to_env}/bin/activate {self.env_name} && {self.command}"

        # TODO: Understand the subprocess.run parameters "shell" and "executable"
        return subprocess.run(
            full_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )


class Venv(EnvManager):
    def run_in_env(
            self
    ) -> subprocess.CompletedProcess:
        
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
            env_manager: EnvManager
    ) -> None:
        
        self.id = simulation_id
        self.run_script_directory = run_script_directory
        self.env_manager = env_manager
    
    # TODO: Should there be an option to supress output files?
    def run(
            self,
            parameters: Dict[str,float]
    ) -> SimulationResult:
        
        result = self.env_manager.run_in_env()

        stdout_file = os.path.join(self.run_script_directory, self.id, ".out")
        stderr_file = os.path.join(self.run_script_directory, self.id, ".err")

        with open(stdout_file, "w") as f_out:
            f_out.write(result.stdout)

        with open(stderr_file, "w") as f_err:
            f_err.write(result.stderr)

        return SimulationResult(
            simulation_id=self.id,
            parameters=parameters,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            return_code=result.returncode
        )


class MockSimulationConfig(BaseModel):
    stdout_file: str = "test_stdout.txt"
    stderr_file: str = "test_stderr.txt"
    return_code: int = 0


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
            parameters: Dict[str,float],
            run_script_directory: Optional[str] = None,
            run_script_filename: Optional[str] = None
    ) -> SimulationResult:
        
        print(f"Running test simulation with parameters: {parameters} and config: {self.config.model_dump()}")

        return SimulationResult(
            simulation_id=simulation_id,
            parameters=parameters,
            stdout_file=self.config.stdout_file,
            stderr_file=self.config.stderr_file,
            return_code=self.config.return_code
        )


class LocalVerosConfig(BaseModel):
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
        command = f"cd {run_script_directory} && {self.config.veros_path} run {gpu_string} --float-type {self.config.float_type} {run_script}"
        return command

    def _edit_run_script(
            self,
            run_script: str,
            parameters: Dict[str,float]
    ) -> None:
        with open(run_script, 'r') as file:
            data = file.readlines()

        # TODO: This is not robust. Need to figure out how to handle the indentation.
        # TODO: How to introduce new parameters that are not already in the setup file?
        # TODO: Check if the parameters are already overwritten in the setup file.
        for i, line in enumerate(data):
            for key, value in parameters.items():
                if line.startswith(f"        settings.{key} ="):
                    print(f"Overwriting {key} in setup file with value: {value}")
                    old_line = data[i].strip()
                    data[i] = f"        settings.{key} = {value}  # default {old_line}\n"
                    break

        with open(run_script, 'w') as file:
            file.writelines(data)

    def set_up_and_run(
            self,
            simulation_id: str,
            parameters: Dict[str,float],
            run_script_directory: str,
            run_script_filename: str
    ) -> SimulationResult:
        run_script = os.path.join(run_script_directory, f"{run_script_filename}.py")
        self._edit_run_script(run_script, parameters) if not self.config.keep_old_params else None
        command = self._make_command(run_script, run_script_directory) if self.config.command is None else self.config.command

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
            env_manager=env_manager)
        result = simulation.run(parameters=parameters)
        return result
