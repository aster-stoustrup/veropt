from tests.test_optimiser import _build_matern_optimiser_ucb
from veropt.optimiser.practice_objectives import Hartmann

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
