from veropt.optimiser.constructors import gpytorch_model


def test_gpytorch_model() -> None:

    # TODO: Make tests. Some with correct input and some with wrong

    model = gpytorch_model(
        n_variables=4,
        n_objectives=2,
        kernels='matern',
        kernel_settings={
            'lengthscale_upper_bound': 5.0,
        },
        training_settings={
            'max_iter': 5_000
        }
    )

    assert False
