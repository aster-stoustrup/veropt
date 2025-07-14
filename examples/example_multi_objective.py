from veropt.optimiser.practice_objectives import DTLZ1, VehicleSafety
from veropt.optimiser.constructors import bayesian_optimiser


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
            'max_iter': 500  # This is just to develop faster, probably not enough to train well
        }
    },
    acquisition_optimiser={
        'optimiser': 'dual_annealing',
        'optimiser_settings': {
            'max_iter': 300
        }
    }
)
