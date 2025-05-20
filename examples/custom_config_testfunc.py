from veropt import BayesOptimiser
from veropt.obj_funcs.predefined_test_functions import *
from veropt.gui import veropt_gui
from veropt.acq_funcs import *
from veropt.kernels import *

n_init_points = 24
n_bayes_points = 64

n_evals_per_step = 4

obj_func = PredefinedTestFunction("Hartmann")
# obj_func = PredefinedFitTestFunction("sine_sum")
# obj_func = PredefinedFitTestFunction("sine_1param")
# obj_func = PredefinedFitTestFunction("sine_2params_offset")
# obj_func = PredefinedFitTestFunction("sine_3params")

n_objs = obj_func.n_objs

acq_func = PredefinedAcqFunction(
    bounds=obj_func.bounds,
    n_objs=n_objs,
    n_evals_per_step=n_evals_per_step,
    acqfunc_name="UCB",
    beta=3.0,
    seq_dist_punish=True,
    alpha=0.7
)

kernel = BayesOptModel(
    n_params=obj_func.n_params,
    n_objs=n_objs,
    model_class_list=[MaternModelBO],
    init_max_iter=10000,
    using_priors=False
)


optimiser = BayesOptimiser(
    n_init_points=n_init_points,
    n_bayes_points=n_bayes_points,
    obj_func=obj_func,
    acq_func=acq_func,
    model=kernel,
    n_evals_per_step=n_evals_per_step
)


# optimiser.run_all_opt_steps()

veropt_gui.run(optimiser)
