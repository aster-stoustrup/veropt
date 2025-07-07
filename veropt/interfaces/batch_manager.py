from abc import ABC, abstractmethod
from enum import StrEnum
import os
import tqdm
import time
import subprocess
from typing import TypeVar, Generic, List, Dict, Literal, Union, Tuple, Optional
from pydantic import BaseModel
from veropt.interfaces.simulation import SimulationResult, SimulationRunner, SimulationResultsDict
from veropt.interfaces.experiment_utility import ExperimentalState, Point, PathManager
from veropt.interfaces.utility import Config, create_directory, copy_files


SR = TypeVar("SR", bound=SimulationRunner)
ConfigType = TypeVar("ConfigType", bound=BaseModel)


def get_job_status_dict(
        output: str
) -> dict[str, str]:
    job_status_dict = {}
    for item in output.split():
        key, *value = item.split('=')
        job_status_dict[key] = value[0] if value else None

    return job_status_dict


def check_if_job_completed(
        job_status_dict: dict,
        error: str
) -> bool:
    completed = False
    
    if job_status_dict['JobState'] == "COMPLETED":
        completed = True

    elif job_status_dict['JobState'] == "COMPLETING":
        completed = True
    
    elif "slurm_load_jobs error: Invalid job id specified" in error:
        completed = True  # TODO: IF RESUBMITTING, THIS IS WRONG!!!

    return completed
    

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
    results_directory: str
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
            result_i_directory = os.path.join(
                self.config.results_directory,
                simulation_id
                )

            create_directory(path=result_i_directory)

            copy_files(
                source_directory=self.config.run_script_root_directory,
                destination_directory=result_i_directory
                )

            experimental_state.points[i].state = "Simulation started"
            result = self.simulation_runner.save_set_up_and_run(
                simulation_id=simulation_id,
                parameters=parameters,
                run_script_directory=result_i_directory,
                run_script_filename=self.config.run_script_filename,
                output_filename=self.config.output_filename
                )
            experimental_state.points[i].state = "Simulation completed"

            experimental_state.points[i].result = result

            results[i] = result

            experimental_state.save_to_json(experimental_state.state_json)

        return results


class LocalSlurmBatchManagerConfig(Config):
    run_script_filename: str
    run_script_root_directory: str
    results_directory: str
    output_filename: str
    check_job_status_sleep_time: int


class LocalSlurmBatchManager(BatchManager):
    def __init__(
            self,
            simulation_runner: SR,
            config: LocalSlurmBatchManagerConfig
    ):
        self.simulation_runner = simulation_runner
        self.config = config

    def set_up_run(
            self,
            i: int
    ) -> tuple[str, str]:
        
        simulation_id = PathManager.make_simulation_id(i=i)
        result_i_directory = os.path.join(
            self.config.results_directory,
            simulation_id
            )

        create_directory(path=result_i_directory)

        copy_files(
            source_directory=self.config.run_script_root_directory,
            destination_directory=result_i_directory
            )
        
        return simulation_id, result_i_directory

    def submit_job(
            self,
            parameters: dict[str, float],
            simulation_id: str,
            result_i_directory: str
    ) -> tuple[Optional[int], SimulationResult]:
        
        result = self.simulation_runner.save_set_up_and_run(
                simulation_id=simulation_id,
                parameters=parameters,
                run_script_directory=result_i_directory,
                run_script_filename=self.config.run_script_filename,
                output_filename=self.config.output_filename
                )
        
        with open(result.stdout_file, "r") as file:
            output = file.read()
        
        if os.path.getsize(result.stderr_file) == 0 and output.strip().isdigit():
            job_id = int(output.strip())
        else:
            print(f"Submission of simulation {result.simulation_id} failed.")
            print("Maximum retries limit reached. Proceeding without resubmission.")
            job_id = None
        
        return job_id, result
    
    def check_job_status(
            self,
            job_id: int,
            state: str
    ) -> str:
        
        pipe = subprocess.Popen(f"scontrol show job {job_id}",
                                shell=True,
                                executable="/bin/bash",
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                text=True)
        
        stdout = pipe.stdout.read()
        stderr = pipe.stderr.read()

        if stdout:
            job_status_dict = get_job_status_dict(output=stdout)
            print("Job {jd[JobId]}/{jd[JobName]} status: {jd[JobState]} (Reason: {jd[Reason]}).".format(jd=job_status_dict))

            completed = check_if_job_completed(
                job_status_dict=job_status_dict,
                error=stderr
                )

            if completed:
                print(f"{job_id} status: COMPLETED")
                state = "Simulation completed"
            
            elif job_status_dict["JobState"] == "RUNNING":
                print(f"{job_id} status: RUNNING")
                state = "Simulation running"


        elif stderr and "slurm_load_jobs error: Invalid job id specified" not in stderr:
            print(f"Error checking job {job_id}: {stderr}")
            print(f"Continuing in 60 seconds.")  # TODO: Move to config; what name?
            time.sleep(60)

        return state


    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:
        
        results = {}

        for i, parameters in dict_of_parameters.items():
            simulation_id, result_i_directory = self.set_up_run(i=i)
            job_id, result = self.submit_job(
                parameters=parameters,
                simulation_id=simulation_id,
                result_i_directory=result_i_directory
                )

            results[i] = result

            experimental_state.points[i].state = "Simulation started" if job_id is not None \
                else "Simulation failed to start"
            experimental_state.points[i].job_id = job_id
            experimental_state.points[i].result = result
        
        experimental_state.save_to_json(experimental_state.state_json)

        pending_jobs = 0
        points = experimental_state.points
        submitted_points = [i for i in points
                            if points[i].state == "Simulation running"
                            or points[i].state == "Simulation started"]
        submitted_jobs = [points[i].job_id for i in points]

        for i in range(len(submitted_jobs)):
            pending_jobs |= (1 << i)

        while pending_jobs:
            for i in range(len(submitted_jobs)):
                point_id, job_id = submitted_points[i], submitted_jobs[i]

                state = self.check_job_status(
                    job_id=job_id,
                    state=experimental_state.points[point_id].state)
                experimental_state.points[point_id].state = state
                if state == "Simulation completed":
                    pending_jobs &= ~(1 << i)
                else:
                    continue

            if pending_jobs:
                print("\nThe following jobs are still pending or running: ")
                for i in range(len(submitted_jobs)):
                    if pending_jobs & (1 << i):
                        print(f"Point {submitted_points[i]}, Slurm Job ID {submitted_jobs[i]}")

                experimental_state.save_to_json(experimental_state.state_json)

                for i in tqdm.tqdm(range(self.config.check_job_status_sleep_time), "Time until next server poll"):
                    time.sleep(1)

        experimental_state.save_to_json(experimental_state.state_json)

        return results


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
