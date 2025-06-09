from abc import ABC, abstractmethod
import os
import shutil
from typing import TypeVar, Generic, List, Dict, Literal
from pydantic import BaseModel
from veropt.interfaces.simulation import SimulationResult, SimulationRunner


SR = TypeVar("SR", bound=SimulationRunner)
ConfigType = TypeVar("ConfigType", bound=BaseModel)


def create_directories(
        path: str,
        names: List[str]
) -> None:
    
    for name in names:

        full_path = os.path.join(path, name)

        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            print(f"Created directory: {full_path}")

        else:
            print(f"Directory already exists: {full_path}")


def copy_files(
        source_directory: str, 
        destination_directory: str
) -> None:
    
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory, exist_ok=True)

    for file_name in os.listdir(source_directory):
        source_file = os.path.join(source_directory, file_name)
        destination_file = os.path.join(destination_directory, file_name)

        if os.path.isfile(source_file):
            if not os.path.exists(destination_file):
                shutil.copy(source_file, destination_directory)
                print(f"Copied {source_file} to {destination_directory}")

            else:
                print(f"File already exists: {destination_file}")

        else:
            print(f"Skipping non-file: {source_file}")


class BatchManager(ABC, Generic[SR, ConfigType]):
    def __init__(
            self,
            simulation_runner: SR,
            config: ConfigType
    ) -> None:
        self.simulation_runner = simulation_runner
        self.config = config

    @abstractmethod
    def run_batch(
            self,
            list_of_parameters: List[dict]
    ) -> List[SimulationResult]:
        ...


class BatchManagerFactory:
    @staticmethod
    def make_batch_manager_config(
        experiment_mode: Literal["local", "local_slurm", "remote_slurm"]
    ) -> ConfigType:
        raise NotImplementedError

    @staticmethod
    def make_batch_manager(
        experiment_mode: Literal["local", "local_slurm", "remote_slurm"], 
        simulation_runner: SR,
        config: ConfigType
    ) -> BatchManager:
        
        if experiment_mode == "local":

            assert isinstance(config, LocalBatchManagerConfig)

            return LocalBatchManager(
                simulation_runner=simulation_runner,
                config=config
            )
        
        elif experiment_mode == "local_slurm":

            assert isinstance(config, LocalSlurmBatchManagerConfig)

            return LocalSlurmBatchManager(             
                simulation_runner=simulation_runner,
                config=config
            )
        
        elif experiment_mode == "remote_slurm":

            assert isinstance(config, RemoteSlurmBatchManagerConfig)

            return RemoteSlurmBatchManager(             
                simulation_runner=simulation_runner,
                config=config
            )


class LocalBatchManagerConfig(BaseModel):
    experiment_id: str
    experiment_root: str
    experiment_directory: str
    latest_point: int
    max_workers: int


# TODO: should latest_point be read from the directory structure?
class LocalBatchManager(BatchManager):
    def _make_point_ids(
            self
    ) -> List[str]:
        
        start = self.config.latest_point + 1
        end = start + self.config.max_workers

        return [f"point={p}" for p in range(start, end)]


    def run_batch(
            self,
            list_of_parameters: List[dict]
    ) -> List[SimulationResult]:
        
        results = []
        point_ids = self._make_point_ids()
        directory_names = [os.path.join("results", id) for id in point_ids]

        create_directories(path=self.config.experiment_directory, names=directory_names)

        for parameters, id in zip(list_of_parameters, point_ids):

            setup_path = os.path.join(self.config.experiment_directory, "results", id)

            copy_files(
                source_directory=self.config.experiment_root,
                destination_directory=setup_path
            )
            # TODO: how should the information about the setup in experiment_rootdir be passed?
            result = self.simulation_runner.save_set_up_and_run(
                simulation_id=id,
                parameters=parameters,
                setup_path=setup_path,
                setup_name=self.config.experiment_id)
            
            # TODO: How to assert that list entries are SimulationResults?
            assert isinstance(result, (SimulationResult, list))

            results.extend(result if isinstance(result, list) else [result])

        return results


class LocalSlurmBatchManagerConfig(BaseModel):
    ...


class LocalSlurmBatchManager(BatchManager):
    def run_batch(
            self,
            list_of_parameters: List[dict]
    ) -> List[SimulationResult]:
        # TODO: Implement
        raise NotImplementedError
    

class RemoteSlurmBatchManagerConfig(BaseModel):
    ...
    

class RemoteSlurmBatchManager(BatchManager):
    def run_batch(
            self,
            list_of_parameters: List[dict]
    ) -> List[SimulationResult]:
        # TODO: Implement
        raise NotImplementedError
