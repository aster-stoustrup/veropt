from veropt import bayesian_optimiser
from veropt.optimiser.practice_objectives import VehicleSafety

objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=36,
    n_evaluations_per_step=4,
    objective=objective,
    model={
        'kernels': 'matern',
        'kernel_settings': {
            'lengthscale_upper_bound': 5.0
        },
        'training_settings': {
            'max_iter': 15_000
        }
    },
    acquisition_function={
        'function': 'qlogehvi',
    },
    acquisition_optimiser={
        'optimiser': 'dual_annealing',
        'optimiser_settings': {
            'max_iter': 500
        },
        'allow_proximity_punishment': True,
        'proximity_punish_settings': {
            'alpha': 0.5,
            'omega': 0.9
        }
    },
    normaliser='zero_mean_unit_variance',
    n_points_before_fitting=12,
    initial_points_generator='random'
)
