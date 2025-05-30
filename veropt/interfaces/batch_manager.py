import abc
import os
import shutil
from typing import TypeVar, Generic, List, Dict
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

# TODO: Naming convention: should it be 'src' and 'dst'?
def copy_files(
        source_directory: str, 
        destination_directory: str
) -> None:
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory, exist_ok=True)
    for file_name in os.listdir(source_directory):
        full_file_name = os.path.join(source_directory, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_directory)
            print(f"Copied {full_file_name} to {destination_directory}")
        else:
            print(f"File already exists: {full_file_name}.")


class BatchManager(abc.ABC, Generic[SR, ConfigType]):
    def __init__(
            self, 
            simulation_runner: SR,
            list_of_parameters: List[dict],
            cfg: ConfigType
    ) -> None:
        self.simulation_runner = simulation_runner
        self.list_of_parameters = list_of_parameters
        self.cfg = cfg

    @abc.abstractmethod
    def run_batch(
            self
    ) -> List[SimulationResult]:
        ...


class LocalBatchManagerConfig(BaseModel):
    experiment_id: str
    path_to_experiment: str
    latest_point: int
    max_workers: int


# TODO: previous config was better; building directory structure should be done in the experiment controller
class LocalBatchManager(BatchManager):
    def make_experiment_dir_path(
            self
    ) -> str:
        return os.path.join(self.cfg.path_to_experiment, f"exp_{self.cfg.experiment_id}")
    
    def make_point_ids(
            self
    ) -> List[str]:
        point_range = range(self.cfg.latest_point, self.cfg.latest_point + self.cfg.max_workers)
        return [f"point={p}" for p in point_range]
    
    def get_experiment_rootdir(
            self,
            experiment_dir: str
    ) -> str:
        return os.path.join(experiment_dir, f"{self.cfg.experiment_id}_setup")

    def run_batch(
            self
    ) -> List[SimulationResult]:
        results = []
        experiment_dir = self.make_experiment_dir_path()
        point_ids = self.make_point_ids()
        experiment_rootdir = self.get_experiment_rootdir(experiment_dir)
        create_directories(path=experiment_dir, names=[f"results/{p}" for p in point_ids])
        for params, id in zip(self.list_of_parameters, point_ids):
            dst_dir = os.path.join(experiment_dir, "results", id)
            copy_files(
                source_directory=experiment_rootdir,
                destination_directory=dst_dir
            )
            # TODO: how should the information about the setup in experiment_rootdir be passed?
            result = self.simulation_runner.set_up_and_run(
                id=id, 
                parameters=params,
                setup_path=dst_dir,
                setup_name=self.cfg.experiment_id)
            # TODO: Should unpack if this is a list of results
            results.append(result)
        return results
