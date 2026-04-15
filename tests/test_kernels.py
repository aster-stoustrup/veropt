import pytest
import torch

from veropt import bayesian_optimiser
from veropt.optimiser.kernels import MaternKernel, DoubleMaternKernel
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


def test_noise_settings_stored_on_kernel() -> None:

    custom_noise = 1e-4
    kernel = MaternKernel(
        n_variables=3,
        noise_settings={'noise': custom_noise, 'train_noise': True}
    )

    assert kernel._noise_settings.noise == pytest.approx(custom_noise)
    assert kernel._noise_settings.train_noise is True
    assert kernel.train_noise is True


def test_noise_settings_defaults_when_not_provided() -> None:

    kernel = MaternKernel(n_variables=3)

    assert kernel._noise_settings.noise == pytest.approx(1e-8)
    assert kernel._noise_settings.noise_lower_bound == pytest.approx(1e-8)
    assert kernel._noise_settings.train_noise is False


def test_noise_settings_applied_after_model_initialisation() -> None:

    custom_noise = 1e-4
    kernel = MaternKernel(
        n_variables=2,
        noise_settings={'noise': custom_noise, 'noise_lower_bound': custom_noise}
    )

    var_tensor = torch.rand(4, 2)
    obj_tensor = torch.rand(4, 1)

    kernel.initialise_model_with_data(
        train_inputs=var_tensor,
        train_targets=obj_tensor.squeeze()
    )

    assert kernel.model_with_data is not None
    actual_noise = float(kernel.model_with_data.likelihood.noise.item())
    assert actual_noise == pytest.approx(custom_noise, rel=0.01)


def test_legacy_kernel_state_raises_key_error() -> None:
    """Loading a v1-format state dict (noise inside 'settings') must raise KeyError —
    the schema gate in load_optimiser_from_state is the only supported upgrade path."""

    legacy_state = {
        'n_variables': 2,
        'settings': {
            'lengthscale_lower_bound': 0.1,
            'lengthscale_upper_bound': 2.0,
            'nu': 2.5,
            'noise': 1e-08,
            'noise_lower_bound': 1e-08,
            'train_noise': False,
        },
        'state_dict': {},
        'train_inputs': [],
        'train_targets': [],
    }

    with pytest.raises(KeyError):
        MaternKernel.from_saved_state(legacy_state)


def test_noise_in_kernel_settings_raises_error() -> None:
    """Passing noise fields inside kernel_settings must raise an AssertionError via _validate_typed_dict."""

    from veropt.optimiser.constructors import gpytorch_single_model

    with pytest.raises(AssertionError, match="noise"):
        gpytorch_single_model(
            n_variables=2,
            kernel='matern',
            settings={'noise': 1e-4}
        )
