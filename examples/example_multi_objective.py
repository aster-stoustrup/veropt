from veropt.optimiser.practice_objectives import DTLZ1, VehicleSafety
from veropt.optimiser.constructors import bayesian_optimiser

from veropt.graphical.visualisation import plot_prediction_grid_from_optimiser
# objective = DTLZ1()
objective = VehicleSafety()


# TODO: Remove quickening settings
optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective,
    model={
        'training_settings': {
            'max_iter': 500  # This is just to develop faster, might not be enough to train well
        }
    },
    acquisition_optimiser={
        'optimiser': 'dual_annealing',
        'optimiser_settings': {
            'max_iter': 300
        }
    }
)

for i in range(4):
    optimiser.run_optimisation_step()

optimiser.suggest_candidates()

plot_prediction_grid_from_optimiser(
    optimiser=optimiser
)
