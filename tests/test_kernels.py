from veropt import bayesian_optimiser
from veropt.optimiser.practice_objectives import Hartmann


def test_run_optimisation_step_rq_matern_kernel() -> None:

    # Just an integration test to ensure this kernel runs
    #   - Could probably make this more minimal

    n_initial_points = 4
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
        verbose=False,
        model={
            'training_settings': {
                'max_iter': 50
            },
            "kernels": "rational_quadratic_and_matern",
            "kernel_settings": {
                "alpha_upper_bound": 0.01,
                "alpha_lower_bound": 0.00001,
                "rq_lengthscale_lower_bound": 0.1,
                "rq_lengthscale_upper_bound": 2.0,
                "matern_lengthscale_lower_bound": 0.1,
                "matern_lengthscale_upper_bound": 2.0
            }
        },
        acquisition_optimiser={
            'optimiser': 'dual_annealing',
            'optimiser_settings': {
                'max_iter': 50
            }
        }
    )

    for i in range(3):
        optimiser.run_optimisation_step()
