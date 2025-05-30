if __name__ == '__main__':
    from veropt import BayesOptimiser, BayesExperiment
    from veropt.obj_funcs.predefined_test_functions import *
    from veropt.acq_funcs import *
    from veropt.kernels import *


def run():

    n_init_points = 16
    n_bayes_points = 64

    n_objs = 1

    beta = 3.0
    gamma = 0.01

    n_evals_per_step = 8

    # obj_func = PredefinedFitTestFunction("sine_3params")
    obj_func = PredefinedTestFunction("Hartmann")
    kernel = BayesOptModel(obj_func.n_params, n_objs, model_class_list=[MaternModelBO], init_max_iter=1000, using_priors=False)

    # alpha_array = np.linspace(0.1, 10, num=20)
    # omega_array = np.linspace(0.1, 1.0, num=10)
    alpha_array = np.linspace(0.1, 10, num=5)
    omega_array = np.linspace(0.1, 1.0, num=5)

    optimiser_list = []

    for alpha in alpha_array:
        omega = 0.5
        acq_func = PredefinedAcqFunction(obj_func.bounds, n_objs, n_evals_per_step=n_evals_per_step,
                                         acqfunc_name="UCB_Var", seq_dist_punish=True, beta=beta, gamma=gamma,
                                         alpha=alpha, omega=omega, )
        optimiser_list.append(BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func, model=kernel,
                                             test_mode=False, using_priors=False, n_evals_per_step=n_evals_per_step,
                                             verbose=False))

    for omega in omega_array:
        alpha = 2.0
        acq_func = PredefinedAcqFunction(obj_func.bounds, n_objs, n_evals_per_step=n_evals_per_step,
                                         acqfunc_name="UCB_Var", seq_dist_punish=True, beta=beta, gamma=gamma,
                                         alpha=alpha, omega=omega)
        optimiser_list.append(BayesOptimiser(n_init_points, n_bayes_points, obj_func, acq_func, model=kernel,
                                             test_mode=False, using_priors=False, n_evals_per_step=n_evals_per_step,
                                             verbose=False))

    parameters = {"alpha": alpha_array, "omega": omega_array}

    experiment = BayesExperiment(optimiser_list, parameters)

    # experiment.run_full_experiment()
    # experiment.run_full_exp_parallel(save=False)
    experiment.save_experiment()


if __name__ == '__main__':
    run()

