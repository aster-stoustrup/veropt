from veropt.optimiser.constructors import gpytorch_model, torch_model_optimiser
from veropt.optimiser.model import GPyTorchFullModel, MaternSingleModel


# TODO: Make tests. Some with correct input and some with wrong


def test_gpytorch_model() -> None:

    n_variables = 4
    n_objectives = 2
    lengthscale_upper_bound = 5.0
    max_iter = 5_000

    single_model_list = []
    for obj_no in range(n_objectives):
        single_model_list.append(
            MaternSingleModel(
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

    assert model.training_settings == model_from_constructors.training_settings
