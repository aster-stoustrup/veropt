from abc import ABC, abstractmethod
from enum import StrEnum
import os
from typing import TypeVar, Generic, List, Dict, Literal, Union, Tuple, Optional
from pydantic import BaseModel
from veropt.interfaces.simulation import SimulationResult, SimulationRunner, SimulationResultsDict
from veropt.interfaces.experiment_utility import ExperimentalState, Point, PathManager
from veropt.interfaces.utility import Config, create_directory, copy_files


SR = TypeVar("SR", bound=SimulationRunner)
ConfigType = TypeVar("ConfigType", bound=BaseModel)


class ExperimentMode(StrEnum):
    LOCAL = "local"
    LOCAL_SLURM = "local_slurm"
    REMOTE_SLURM = "remote_slurm"


class BatchManager(ABC, Generic[SR]):
    def __init__(
            self,
            simulation_runner: SR,
    ):
        self.simulation_runner = simulation_runner

    @abstractmethod
    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:
        ...


class BatchManagerFactory:
    @staticmethod
    def make_batch_manager_config(
            experiment_mode: str,
            run_script_filename: str,
            run_script_root_directory: str,
            output_filename: str
    ) -> BaseModel:

        if experiment_mode == ExperimentMode.LOCAL:

            return LocalBatchManagerConfig(
                run_script_filename=run_script_filename,
                run_script_root_directory=run_script_root_directory,
                output_filename=output_filename
                )

        else:
            raise NotImplementedError

    @staticmethod
    def make_batch_manager(
            experiment_mode: str,
            simulation_runner: SR,
            config: BaseModel
    ) -> BatchManager:

        if experiment_mode == ExperimentMode.LOCAL:

            assert isinstance(config, LocalBatchManagerConfig)

            return LocalBatchManager(
                simulation_runner=simulation_runner,
                config=config
                )

        elif experiment_mode == ExperimentMode.LOCAL_SLURM:

            assert isinstance(config, LocalSlurmBatchManagerConfig)

            return LocalSlurmBatchManager(
                simulation_runner=simulation_runner,
                config=config
                )

        elif experiment_mode == ExperimentMode.REMOTE_SLURM:

            assert isinstance(config, RemoteSlurmBatchManagerConfig)

            return RemoteSlurmBatchManager(
                simulation_runner=simulation_runner,
                config=config
                )

        else:

            raise ValueError(f"Unsupported mode: {experiment_mode!r}")


class LocalBatchManagerConfig(Config):
    run_script_filename: str
    run_script_root_directory: str
    output_filename: str


# TODO: should latest_point be read from the directory structure?
class LocalBatchManager(BatchManager):
    def __init__(
            self,
            simulation_runner: SR,
            config: LocalBatchManagerConfig
    ):
        self.simulation_runner = simulation_runner
        self.config = config

    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:

        results = {}

        # TODO: This is fine for now but could be done better; however it is important to check
        #       - If use Process manager instead of for loop, does Simulation structure have to change?
        #       - Support for running simulation on GPUs!!!
        for i, parameters in dict_of_parameters.items():

            simulation_id = PathManager.make_simulation_id(i=i)
            result_directory = os.path.join(
                experimental_state.experiment_directory,
                "results",
                simulation_id
                )

            create_directory(path=result_directory)

            copy_files(
                source_directory=self.config.run_script_root_directory,
                destination_directory=result_directory
                )

            experimental_state.points[i].state = "Simulation started"
            result = self.simulation_runner.save_set_up_and_run(
                simulation_id=simulation_id,
                parameters=parameters,
                run_script_directory=result_directory,
                run_script_filename=self.config.run_script_filename,
                output_filename=self.config.output_filename)
            experimental_state.points[i].state = "Simulation finished"

            experimental_state.points[i].result = result

            results[i] = result

            experimental_state.save_to_json(experimental_state.state_json)

        return results


class LocalSlurmBatchManagerConfig(Config):
    ...


class LocalSlurmBatchManager(BatchManager):
    def __init__(
            self,
            simulation_runner: SR,
            config: LocalSlurmBatchManagerConfig
    ):
        self.simulation_runner = simulation_runner
        self.config = config

    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:
        # TODO: Implement
        raise NotImplementedError


class RemoteSlurmBatchManagerConfig(Config):
    ...


class RemoteSlurmBatchManager(BatchManager):
    def __init__(
            self,
            simulation_runner: SR,
            config: RemoteSlurmBatchManagerConfig
    ):
        self.simulation_runner = simulation_runner
        self.config = config

    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:
        # TODO: Implement
        raise NotImplementedError
