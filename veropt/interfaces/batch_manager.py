from abc import ABC, abstractmethod
import os
import tqdm
import subprocess
import time
from typing import Optional, Literal, Union

from veropt.interfaces.simulation import SimulationResult, SimulationRunner, SimulationResultsDict
from veropt.interfaces.experiment_utility import (
    ExperimentMode, ExperimentalState, Point, PathManager
)
from veropt.interfaces.utility import create_directory, copy_files


def _check_if_point_exists(
        i: int,
        parameters: dict[str, float],
        experimental_state: ExperimentalState
) -> None:

    try:
        _ = experimental_state.points[i].state
    except KeyError:
        _initialise_point(
            i=i,
            parameters=parameters,
            experimental_state=experimental_state
        )


def _initialise_point(
        i: int,
        parameters: dict[str, float],
        experimental_state: ExperimentalState
) -> None:

    point = Point(
        state="Initialised by batch manager",
        parameters=parameters
    )

    if experimental_state.next_point == i:
        experimental_state.update(point)
    else:
        experimental_state.points[i] = point  # TODO: Warning here?

    print(f"Point {i} not found; initialising with batch manager.")


def _get_job_status_dict(output: str) -> dict[str, Optional[str]]:
    job_status_dict = {}
    for item in output.split():
        key, *value = item.split('=')
        job_status_dict[key] = value[0] if value else None

    return job_status_dict


def _check_if_job_completed(
        job_status_dict: dict,
        error: str
) -> bool:
    completed = False

    if job_status_dict['JobState'] == "COMPLETED":
        completed = True

    elif job_status_dict['JobState'] == "COMPLETING":
        completed = True

    elif "slurm_load_jobs error: Invalid job id specified" in error:
        completed = True  # TODO: IF RESUBMITTING WITH NEW JOB_ID, THIS IS WRONG!!!

    return completed


class BatchManager(ABC):
    def __init__(
            self,
            simulation_runner: SimulationRunner,
            run_script_filename: str,
            run_script_root_directory: str,
            results_directory: str,
            output_filename: str,
            check_job_status_frequency: Optional[int],
            remote: bool = False,
            hostname: Optional[str] = None
    ):
        self.simulation_runner = simulation_runner
        self.run_script_filename = run_script_filename
        self.run_script_root_directory = run_script_root_directory
        self.results_directory = results_directory
        self.output_filename = output_filename
        self.check_job_status_frequency = 60 if check_job_status_frequency is None \
            else check_job_status_frequency
        self.remote = remote
        self.hostname = hostname

    def _set_up_directory(
            self,
            i: int
    ) -> tuple[str, str]:

        simulation_id = PathManager.make_simulation_id(i=i)
        result_i_directory = os.path.join(
            self.results_directory,
            simulation_id
        )

        create_directory(path=result_i_directory)

        copy_files(
            source_directory=self.run_script_root_directory,
            destination_directory=result_i_directory
        )

        return simulation_id, result_i_directory


def _get_batch_manager_class(
        experiment_mode: Literal['local', 'local_slurm', 'remote_slurm'],
) -> type[BatchManager]:
    batch_manager_classes = {
        "local": LocalBatchManager,
        "local_slurm": LocalSlurmBatchManager,
        "remote_slurm": RemoteSlurmBatchManager
    }

    return batch_manager_classes[experiment_mode]


def make_batch_manager(
        experiment_mode: Literal['local', 'local_slurm', 'remote_slurm'],
        simulation_runner: SimulationRunner,
        run_script_filename: str,
        run_script_root_directory: str,
        results_directory: str,
        output_filename: str,
        check_job_status_frequency: int = 60,
        batch_manager_class: Optional[type[BatchManager]] = None
) -> Union['DirectBatchManager', 'SubmitBatchManager']:

    assert experiment_mode in ExperimentMode, \
        f"Unsupported experiment mode: {experiment_mode};" \
        f"expected one of: {[mode.value for mode in ExperimentMode]}."

    remote = True if experiment_mode == "remote_slurm" else False

    batch_manager_class = batch_manager_class or _get_batch_manager_class(experiment_mode)

    return batch_manager_class(  # type: ignore # abstract BatchManager is never initialised here
        simulation_runner=simulation_runner,
        run_script_filename=run_script_filename,
        run_script_root_directory=run_script_root_directory,
        results_directory=results_directory,
        output_filename=output_filename,
        check_job_status_frequency=check_job_status_frequency,
        remote=remote
    )


class DirectBatchManager(BatchManager, ABC):
    @abstractmethod
    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:
        ...


class LocalBatchManager(DirectBatchManager):
    def run_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> SimulationResultsDict:

        results = {}

        for i, parameters in dict_of_parameters.items():
            simulation_id, result_i_directory = self._set_up_directory(i=i)

            _check_if_point_exists(
                i=i,
                parameters=parameters,
                experimental_state=experimental_state
            )

            experimental_state.points[i].state = "Simulation started"
            result = self.simulation_runner.save_set_up_and_run(
                simulation_id=simulation_id,
                parameters=parameters,
                run_script_directory=result_i_directory,
                run_script_filename=self.run_script_filename,
                output_filename=self.output_filename
            )
            experimental_state.points[i].state = "Simulation completed"
            experimental_state.points[i].result = result

            results[i] = result

            experimental_state.save_to_json(experimental_state.state_json)

        return results


class SubmitBatchManager(BatchManager, ABC):

    @abstractmethod
    def submit_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> None:
        ...

    @abstractmethod
    def wait_for_jobs(
            self,
            experimental_state: ExperimentalState
    ) -> None:
        ...


class LocalSlurmBatchManager(SubmitBatchManager):

    # TODO: These slurm specific class methods should be moved to a new SlurmBatchManager superclass if
    #  RemoteSlurmBatchManager needs them
    def _submit_job(
            self,
            parameters: dict[str, float],
            simulation_id: str,
            result_i_directory: str
    ) -> tuple[Optional[int], SimulationResult]:

        result = self.simulation_runner.save_set_up_and_run(
            simulation_id=simulation_id,
            parameters=parameters,
            run_script_directory=result_i_directory,
            run_script_filename=self.run_script_filename,
            output_filename=self.output_filename
        )

        job_id = self._get_job_id(
            result=result
        )

        if job_id is None:
            print(f"Submission of simulation {result.simulation_id} failed.")
            print("Maximum retries limit reached. Proceeding without resubmission.")

        return job_id, result

    def _get_job_id(
            self,
            result: SimulationResult
    ) -> Optional[int]:

        # TODO: Probably need to detect if the stdout is of the expected format?
        #   - And otherwise raise exception so the user will know they need to overwrite this
        #   - Could also leave this as an abstract, but maybe it's a good assumption that this will work for many cases?

        with open(result.stdout_file, "r") as stdout_file:
            output = stdout_file.read()

        if os.path.getsize(result.stderr_file) == 0 and output.strip().isdigit():
            job_id = int(output.strip())
        else:
            job_id = None

        return job_id

    def _check_job_status(
            self,
            job_id: int,
            state: str
    ) -> str:

        command = f"scontrol show job {job_id}"
        command_arguments = command.split(" ")

        process = subprocess.run(
            args=command_arguments,
            capture_output=True,
            text=True
        )

        output, error = process.stdout, process.stderr

        if output:
            job_status_dict = _get_job_status_dict(output=output)
            status_message = "Job {jd[JobId]}/{jd[JobName]} status: {jd[JobState]} (Reason: {jd[Reason]})."
            print(status_message.format(jd=job_status_dict))

            completed = _check_if_job_completed(
                job_status_dict=job_status_dict,
                error=error
            )

            if completed:
                print(f"{job_id} status: COMPLETED")
                state = "Simulation completed"

            elif job_status_dict["JobState"] == "RUNNING":
                print(f"{job_id} status: RUNNING")
                state = "Simulation running"

            elif job_status_dict["JobState"] == "FAILED":
                print(f"{job_id} status: FAILED")
                state = "Simulation failed"  # TODO: Extra check for slurm log file

        elif "slurm_load_jobs error: Invalid job id specified" in error:
            # TODO: Make this nicer
            #  - note, Aster: Made this as a quick fix as current solution wasn't working
            print(f"{job_id} status: COMPLETED")
            state = "Simulation completed"

        elif error and "slurm_load_jobs error: Invalid job id specified" not in error:
            print(f"Error checking job {job_id}: {error}")
            print("Continuing in 60 seconds.")  # TODO: Move to config; what name?
            time.sleep(60)

        return state

    def _check_pending_jobs(
            self,
            experimental_state: ExperimentalState
    ) -> None:

        pending_jobs = 0
        points = experimental_state.points
        submitted_points = [i for i in points if (
            points[i].state == "Simulation running" or points[i].state == "Simulation started"
        )]
        submitted_jobs = [points[i].job_id for i in submitted_points]

        for i in range(len(submitted_jobs)):
            pending_jobs |= (1 << i)

        while pending_jobs:
            for i in range(len(submitted_jobs)):
                point_id, job_id = submitted_points[i], submitted_jobs[i]

                state = self._check_job_status(
                    job_id=job_id,  # type: ignore # job_id cannot be None here (job must be submitted or running)
                    state=experimental_state.points[point_id].state,
                )

                experimental_state.points[point_id].state = state

                if state == "Simulation completed":
                    pending_jobs &= ~(1 << i)
                elif state == "Simulation failed":
                    pending_jobs &= ~(1 << i)
                else:
                    continue

            if pending_jobs:
                print("\nThe following jobs are still pending or running: ")
                for i in range(len(submitted_jobs)):
                    if pending_jobs & (1 << i):
                        print(f"Point {submitted_points[i]}, Slurm Job ID {submitted_jobs[i]}")

                experimental_state.save_to_json(experimental_state.state_json)

                for i in tqdm.tqdm(range(self.check_job_status_frequency), "Time until next server poll"):
                    time.sleep(1)

    def submit_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> None:

        results = {}

        for i, parameters in dict_of_parameters.items():
            simulation_id, result_i_directory = self._set_up_directory(i=i)
            job_id, result = self._submit_job(
                parameters=parameters,
                simulation_id=simulation_id,
                result_i_directory=result_i_directory
            )

            results[i] = result

            _check_if_point_exists(
                i=i,
                parameters=parameters,
                experimental_state=experimental_state
            )

            experimental_state.points[i].state = "Simulation started" if job_id is not None \
                else "Simulation failed to start"
            experimental_state.points[i].job_id = job_id
            experimental_state.points[i].result = result

        experimental_state.save_to_json(experimental_state.state_json)

    def wait_for_jobs(
            self,
            experimental_state: ExperimentalState
    ) -> None:

        self._check_pending_jobs(experimental_state=experimental_state)
        experimental_state.save_to_json(experimental_state.state_json)


class RemoteSlurmBatchManager(SubmitBatchManager):
    def submit_batch(
            self,
            dict_of_parameters: dict[int, dict],
            experimental_state: ExperimentalState
    ) -> None:
        # TODO: Implement
        raise NotImplementedError("Remote slurm experiments are not supported yet.")

    def wait_for_jobs(
            self,
            experimental_state: ExperimentalState
    ) -> None:
        # TODO: Implement
        raise NotImplementedError("Remote slurm experiments are not supported yet.")
