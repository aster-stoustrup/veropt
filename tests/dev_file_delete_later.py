
from veropt.optimiser import bayesian_optimiser
from veropt.optimiser.constructors import botorch_acquisition_function, gpytorch_model

from veropt.optimiser.practice_objectives import Hartmann

n_initial_points = 16
n_bayesian_points = 32

n_evalations_per_step = 4

objective = Hartmann(
    n_variables=6
)


optimiser = bayesian_optimiser(
    n_initial_points=n_initial_points,
    n_bayesian_points=n_bayesian_points,
    n_evaluations_per_step=n_evalations_per_step,
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

from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_state, save_to_json

optimiser.run_optimisation_step()
optimiser.run_optimisation_step()
optimiser.run_optimisation_step()
optimiser.run_optimisation_step()
optimiser.suggest_candidates()

# from veropt.graphical.visualisation import plot_prediction_grid_from_optimiser
#
# plot_prediction_grid_from_optimiser(optimiser)

# optimiser.run_optimisation_step()
# optimiser.run_optimisation_step()
# optimiser.run_optimisation_step()

# save_to_json(
#     object_to_save=optimiser,
#     file_name='test'
# )
#
# reloaded_optimiser = load_optimiser_from_state(
#     file_name='test'
# )


# optimiser = bayesian_optimiser(
#     n_initial_points=n_initial_points,
#     n_bayesian_points=n_bayesian_points,
#     n_evaluations_per_step=n_evalations_per_step,
#     objective=objective,
#     model={
#         'kernels': 'matern',
#         'kernel_settings': {
#             'lengthscale_upper_bound': 5.0
#         },
#         'training_settings': {
#             'max_iter': 15_000
#         }
#     },
#     acquisition_function={
#         'function': 'ucb',
#         'parameters': {
#             'beta': 3.0
#         }
#     },
#     acquisition_optimiser={
#         'optimiser': 'dual_annealing',
#         'proximity_punish_settings':{
#             'alpha': 0.5
#         }
#     },
#     renormalise_each_step=False
# )


# acq_func = botorch_acquisition_function(
#     n_variables=objective.n_variables,
#     n_objectives=objective.n_objectives
# )
#
#
# optimiser = bayesian_optimiser(
#     n_initial_points=n_initial_points,
#     n_bayesian_points=n_bayesian_points,
#     n_evaluations_per_step=n_evalations_per_step,
#     objective=objective,
#     model={
#         'kernels': 'matern',
#         'kernel_settings': {
#             'lengthscale_upper_bound': 5.0
#         },
#         'training_settings': {
#             'max_iter': 15_000
#         }
#     },
#     acquisition_function=acq_func,
#     acquisition_optimiser={
#         'optimiser': 'dual_annealing',
#         'proximity_punish_settings':{
#             'alpha': 0.5
#         }
#     },
#     renormalise_each_step=False
# )

# model = gpytorch_model(
#     n_variables=4,
#     n_objectives=2,
#     kernels='matern',
#     kernel_settings={
#         'lengthscale_upper_bound': 5.0,
#     },
#     training_settings={
#         'max_iter': 5_000
#     }
# )

# for i in range (5):
#     optimiser.run_optimisation_step()
