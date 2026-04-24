from veropt.optimiser.constructors import gpytorch_model, torch_model_optimiser
from veropt.optimiser.kernels import MaternKernel
from veropt.optimiser.model import GPyTorchFullModel


# TODO: Make tests. Some with correct input and some with wrong


def test_gpytorch_model() -> None:

    n_variables = 4
    n_objectives = 2
    lengthscale_upper_bound = 5.0
    max_iter = 5_000

    single_model_list = []
    for obj_no in range(n_objectives):
        single_model_list.append(
            MaternKernel(
                n_variables=n_variables,
                lengthscale_upper_bound=lengthscale_upper_bound
            )
        )

    model = GPyTorchFullModel.from_the_beginning(
        n_variables=n_variables,
        n_objectives=n_objectives,
        single_model_list=single_model_list,
        model_optimiser=torch_model_optimiser(),
        max_iter=max_iter
    )

    model_from_constructors = gpytorch_model(
        n_variables=n_variables,
        n_objectives=n_objectives,
        kernels='matern',
        kernel_settings={
            'lengthscale_upper_bound': lengthscale_upper_bound,
        },
        training_settings={
            'max_iter': max_iter
        }
    )

    for obj_no in range(n_objectives):

        assert model._model_list[obj_no].get_settings() == model_from_constructors._model_list[obj_no].get_settings()

        class_name = model._model_list[obj_no].__class__.__name__
        class_name_from_constructors = model_from_constructors._model_list[obj_no].__class__.__name__

        assert class_name == class_name_from_constructors

    assert model.settings == model_from_constructors.settings


def test_noise_settings_single_kernel_single_noise() -> None:
    """A single kernel with a single noise_settings dict applies to all objectives."""

    n_variables = 3
    n_objectives = 2
    custom_noise = 1e-4

    model = gpytorch_model(
        n_variables=n_variables,
        n_objectives=n_objectives,
        noise_settings={'noise': custom_noise, 'noise_lower_bound': custom_noise}
    )

    for objective_no in range(n_objectives):
        kernel = model._model_list[objective_no]
        assert kernel._noise_settings.noise == custom_noise
        assert kernel._noise_settings.noise_lower_bound == custom_noise


def test_noise_settings_broadcast_across_list_of_kernels() -> None:
    """A list of kernels with a single noise_settings dict applies the same noise to all."""

    n_variables = 3
    n_objectives = 2
    custom_noise = 5e-5

    model = gpytorch_model(
        n_variables=n_variables,
        n_objectives=n_objectives,
        kernels=['matern', 'matern'],  # type: ignore[arg-type]
        noise_settings={'noise': custom_noise}
    )

    for objective_no in range(n_objectives):
        kernel = model._model_list[objective_no]
        assert kernel._noise_settings.noise == custom_noise


def test_noise_settings_per_objective() -> None:
    """A list of noise_settings with one dict per objective gives each kernel its own noise."""

    n_variables = 3
    n_objectives = 2
    noise_per_objective = [1e-4, 5e-5]

    model = gpytorch_model(
        n_variables=n_variables,
        n_objectives=n_objectives,
        kernels=['matern', 'matern'],  # type: ignore[arg-type]
        noise_settings=[
            {'noise': noise_per_objective[0]},
            {'noise': noise_per_objective[1]},
        ]
    )

    for objective_no in range(n_objectives):
        kernel = model._model_list[objective_no]
        assert kernel._noise_settings.noise == noise_per_objective[objective_no]


def test_noise_settings_list_wrong_length_raises() -> None:
    """Providing a noise_settings list with wrong length must raise an AssertionError."""

    import pytest

    with pytest.raises(AssertionError, match="noise_settings"):
        gpytorch_model(
            n_variables=3,
            n_objectives=2,
            kernels=['matern', 'matern'],  # type: ignore[arg-type]
            noise_settings=[{'noise': 1e-4}]  # length 1, but n_objectives=2
        )