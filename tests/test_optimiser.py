import torch

from veropt.optimiser.normalisation import NormaliserZeroMeanUnitVariance
from veropt.optimiser.objective import CallableObjective
from veropt.optimiser.practice_objectives import Hartmann
from .test_prediction import _build_matern_predictor_qlogehvi, _build_matern_predictor_ucb
from veropt.optimiser.optimiser import BayesianOptimiser


def _build_matern_optimiser_ucb(
        n_initial_points: int,
        n_bayesian_points: int,
        n_evaluations_per_step: int,
        objective: CallableObjective
) -> BayesianOptimiser:

    predictor = _build_matern_predictor_ucb(
        bounds=objective.bounds,
        n_variables=objective.n_variables,
        n_objectives=objective.n_objectives,
        n_evaluations_per_step=n_evaluations_per_step
    )

    return BayesianOptimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evaluations_per_step,
        objective=objective,
        predictor=predictor,
        normaliser_class=NormaliserZeroMeanUnitVariance
    )


def test_run_optimisation_step() -> None:
    n_initial_points = 16
    n_bayesian_points = 32

    n_evalations_per_step = 4

    objective = Hartmann(
        n_variables=6
    )

    optimiser = _build_matern_optimiser_ucb(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evalations_per_step,
        objective=objective
    )

    for i in range (4):
        optimiser.run_optimisation_step()

    # TODO: Mostly wanna see if this runs
    #   - Maybe we can figure out something useful to test, otherwise can just start with something that checks
    #    that it's not failing?
    #   - Idea: make very "obvious" opt problem and check suggested bayes point is within range of expected position?
    #       - Though as an integration test, I guess it'll be harder to control all points...?
    #   - Extra idea: Make tests with controlled seed (if possible) so we can test consistency across PR's


def test__set_up_settings_nans() -> None:
    # TODO: Implement
    #   - Not sure about current implementation of handling nans
    #   - Might need to use as a context manager during training?

    assert False
