# TODO: Fix this on a general level
#   - Consider if this is a good fix or if we'd like to try to change the np numbers to float32
import torch
torch.set_default_dtype(torch.float64)

# import matplotlib
# matplotlib.use('WebAgg')

from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.acq_funcs import *
from veropt.kernels import *
from veropt.gui import veropt_gui

n_init_points = 16
n_bayes_points = 64

# TODO: Look into bug at = 1
n_evals_per_step = 4


# obj_func = PredefinedTestFunction("BraninCurrin")
# obj_func = PredefinedTestFunction("VehicleSafety")
obj_func = PredefinedTestFunction("DTLZ1")
# obj_func = PredefinedTestFunction("DTLZ1", n_params=12, n_objs=10)

n_objs = obj_func.n_objs

acq_func = PredefinedAcqFunction(
    bounds=obj_func.bounds,
    n_objs=n_objs,
    n_evals_per_step=n_evals_per_step,
    acqfunc_name='qLogEHVI',
    seq_dist_punish=True,
    alpha=1.0,
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
    normalise=True
)

for i in range(n_init_points//n_evals_per_step + 1):
    optimiser.run_opt_step()

# optimiser.plot_prediction(0, 1)

# veropt_gui.run(optimiser)
