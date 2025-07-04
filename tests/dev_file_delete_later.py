# TODO: Fix in project
import torch

from veropt.optimiser import bayesian_optimiser
from veropt.optimiser.constructors import botorch_acquisition_function, gpytorch_model

torch.set_default_dtype(torch.float64)

from tests.test_optimiser import _build_matern_optimiser_ucb
from veropt.optimiser.practice_objectives import Hartmann

n_initial_points = 16
n_bayesian_points = 32

n_evalations_per_step = 4

objective = Hartmann(
    n_variables=6
)

# optimiser = _build_matern_optimiser_ucb(
#     n_initial_points=n_initial_points,
#     n_bayesian_points=n_bayesian_points,
#     n_evaluations_per_step=n_evalations_per_step,
#     objective=objective
# )

# TODO: Make tests for various calls to this and sub-functions
#   - Probably especially really smart to see if it fails in all the right ways
#       - I.e. what if I call it with prox punish settings but allow=False
#       - And do we get told the options for anything if we e.g. misspell
optimiser = bayesian_optimiser(
    n_initial_points=n_initial_points,
    n_bayesian_points=n_bayesian_points,
    n_evaluations_per_step=n_evalations_per_step,
    objective=objective
)

from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_json, save_to_json

optimiser.run_optimisation_step()
optimiser.run_optimisation_step()

save_to_json(
    object_to_save=optimiser,
    file_name='test'
)

reloaded_optimiser = load_optimiser_from_json(
    file_name='test'
)


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
