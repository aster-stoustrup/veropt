from veropt.optimiser.practice_objectives import Hartmann
from veropt import bayesian_optimiser

objective = Hartmann(
    n_variables=6
)

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective
)
