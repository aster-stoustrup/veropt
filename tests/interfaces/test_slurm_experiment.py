import tempfile
from typing import Optional

from veropt.interfaces.batch_manager import FakeSubmitBatchManager
from veropt.interfaces.experiment import Experiment
from veropt.interfaces.experiment_utility import ExperimentConfig
from veropt.interfaces.local_simulation import MockSimulationConfig, MockSimulationRunner
from veropt.interfaces.result_processing import MockResultProcessor


# --- Helpers ---

def _make_experiment_config(
        path_to_experiment: str,
        version: Optional[str] = None
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test_slurm",
        version=version,
        parameter_names=["param1"],
        parameter_bounds={"param1": [0.0, 1.0]},
        path_to_experiment=path_to_experiment,
        experiment_mode="local_slurm",  # type: ignore[arg-type]  # pydantic casts string internally
        output_filename="output"
    )


def _make_result_processor() -> MockResultProcessor:
    # fixed_objective=False so the processor reads the output file written by FakeSubmitBatchManager.
    # Each point's file contains float(point_no + 1), giving unique per-point objective values.
    return MockResultProcessor(
        objective_names=["obj1"],
        objectives={"obj1": 1.0},
        fixed_objective=False
    )


def _make_optimiser_config() -> dict:
    # Small iteration counts so tests run quickly
    return dict(
        n_initial_points=4,
        n_bayesian_points=4,
        n_evaluations_per_step=4,
        verbose=False,
        model={"training_settings": {"max_iter": 5}},
        acquisition_optimiser={
            "optimiser": "dual_annealing",
            "optimiser_settings": {"max_iter": 5}
        }
    )


def _make_simulation_runner() -> MockSimulationRunner:
    return MockSimulationRunner(config=MockSimulationConfig())


# --- Tests ---

def test_submit_experiment_submits_expected_batches() -> None:
    """A fresh submit-mode experiment should run end-to-end and submit the right number of batches."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(),
            result_processor=_make_result_processor(),
            experiment_config=_make_experiment_config(tmp_dir),
            optimiser_config=_make_optimiser_config(),
            batch_manager_class=FakeSubmitBatchManager
        )

        experiment.run_experiment()

        fake_batch_manager = experiment.batch_manager
        assert isinstance(fake_batch_manager, FakeSubmitBatchManager)

        # n_total_steps = (4 + 4) // 4 + 1 = 3
        # Step 0 submits initial batch, step 1 submits bayesian batch, step 2 is final (no submission)
        assert fake_batch_manager.n_batches_submitted == 2
        assert fake_batch_manager.n_jobs_submitted == 8
        assert experiment.n_points_evaluated == 8


def test_new_version_experiment_processes_pending_jobs() -> None:
    """
    After continue_with_new_version, running the experiment should correctly wait for and
    process the pending jobs that were submitted by the old experiment.

    Scenario:
    - Old experiment runs 2 steps: step 0 submits 4 initial points, step 1 processes initial
      results and submits 4 bayesian points. The bayesian jobs are left pending.
    - New version rebuilds from the 4 evaluated initial points (n_steps_to_reevaluate = 1).
    - Running the new experiment (1 remaining step) should wait for the pending bayesian jobs,
      process them, and feed results to the optimiser.

    This test exposes the bug where just_rebuilt is never reset to False, causing the
    wait_for_jobs / result processing block to be skipped on all subsequent steps.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        old_config = _make_experiment_config(tmp_dir, version=None)
        new_config = _make_experiment_config(tmp_dir, version="v2")
        optimiser_config = _make_optimiser_config()

        # --- Build and partially run old experiment ---

        old_experiment = Experiment.from_the_beginning(
            simulation_runner=_make_simulation_runner(),
            result_processor=_make_result_processor(),
            experiment_config=old_config,
            optimiser_config=optimiser_config,
            batch_manager_class=FakeSubmitBatchManager
        )

        old_experiment.run_experiment_step()  # Step 0: submits 4 initial points
        old_experiment.run_experiment_step()  # Step 1: processes initial, submits 4 bayesian

        # Sanity-check: old experiment has submitted 8 points but only evaluated 4 (initial)
        assert old_experiment.n_points_submitted == 8
        assert old_experiment.n_points_evaluated == 4

        # Bayesian points are still waiting
        assert old_experiment.state.points[4].state == "Simulation started"

        # --- Create new version, which rebuilds from the 4 initial evaluated points ---

        new_experiment = Experiment.continue_with_new_version(
            simulation_runner=_make_simulation_runner(),
            result_processor=_make_result_processor(),
            old_experiment_config=old_config,
            new_experiment_config=new_config,
            optimiser_config=optimiser_config,
            batch_manager_class=FakeSubmitBatchManager
        )

        # After rebuild: 4 initial points re-evaluated, old state (8 submitted) carried over
        assert new_experiment.n_points_evaluated == 4
        assert new_experiment.n_points_submitted == 8
        assert new_experiment.just_rebuilt is True

        # --- Run the 1 remaining step of the new experiment ---
        # Correct behaviour: wait for points 4-7, process their results, feed to optimiser
        # Bug behaviour: just_rebuilt never resets → wait_for_jobs and result processing are skipped

        new_experiment.run_experiment()

        # The bayesian jobs from the old experiment should have been waited for
        assert new_experiment.state.points[4].state == "Simulation completed"
        assert new_experiment.state.points[7].state == "Simulation completed"

        # All 8 points (4 initial + 4 bayesian) should now be in the optimiser
        assert new_experiment.n_points_evaluated == 8

        # Data integrity: objective value for point N is float(N + 1), so the last 4 evaluated
        # points should be 5.0, 6.0, 7.0, 8.0 (not a repeat of the initial 1.0–4.0)
        evaluated_objectives = new_experiment.optimiser.evaluated_objectives_real_units
        last_four_objectives = evaluated_objectives[-4:, 0].tolist()
        assert last_four_objectives == [5.0, 6.0, 7.0, 8.0]

