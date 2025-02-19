from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.acq_funcs import *
from veropt.kernels import *
from veropt.gui import veropt_gui

# n_init_points = 16
n_init_points = 16 * 4
n_bayes_points = 64

n_evals_per_step = 4
points_before_fitting = n_init_points - n_evals_per_step * 3


obj_func = PredefinedTestFunction("BraninCurrin")
# obj_func = PredefinedTestFunction("VehicleSafety")
# obj_func = PredefinedTestFunction("DTLZ1")
# obj_func = PredefinedTestFunction("DTLZ2")
# obj_func = PredefinedTestFunction("DTLZ2", n_params=7, n_objs=4)

n_objs = obj_func.n_objs

acq_func = PredefinedAcqFunction(
    bounds=obj_func.bounds,
    n_objs=n_objs,
    n_evals_per_step=n_evals_per_step,
    acqfunc_name='qLogEHVI',
    seq_dist_punish=True,
    alpha=1.2,
    omega=1.0
)


constraints = {
    "covar_module": {
        "raw_lengthscale": [0.1, 2.0]}
}

model_list = n_objs * [MaternModelBO]  # Matern is the default

kernel = BayesOptModel(
    n_params=obj_func.n_params,
    n_objs=obj_func.n_objs,
    model_class_list=model_list,
    constraint_dict_list=constraints
)

optimiser = BayesOptimiser(
    n_init_points=n_init_points,
    n_bayes_points=n_bayes_points,
    obj_func=obj_func,
    acq_func=acq_func,
    model=kernel,
    n_evals_per_step=n_evals_per_step,
    normalise=True,
    points_before_fitting=points_before_fitting
)


# for i in range(n_init_points//n_evals_per_step + 1):
#     optimiser.run_opt_step()

# optimiser.plot_prediction(0, 1)

veropt_gui.run(optimiser)

# for i in range(n_init_points // n_evals_per_step):
#     optimiser.run_opt_step()

# optimiser.suggest_opt_steps()

# from veropt.visualisation import *

# run_prediction_grid_app(optimiser)
# plot_prediction_grid_from_optimiser(optimiser)
