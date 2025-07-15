from veropt.graphical.visualisation import plot_prediction_grid_from_optimiser
from veropt import bayesian_optimiser
from veropt.optimiser.practice_objectives import VehicleSafety

objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective
)

for i in range(4):
    optimiser.run_optimisation_step()

optimiser.suggest_candidates()

plot_prediction_grid_from_optimiser(
    optimiser=optimiser
)
