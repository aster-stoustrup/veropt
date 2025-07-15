import pytest
import torch

from veropt.optimiser.constructors import botorch_predictor


def test_botorch_predict_values_1_objective() -> None:
    bounds = torch.tensor([-10.0, 10.0])

    variable_1_array = torch.arange(
        start=float(bounds[0]),
        end=float(bounds[1]),
        step=0.1
    )
    variable_1_array = variable_1_array.unsqueeze(1)

    variable_values = torch.tensor([[1.2, 3.2, 2.1, 5.1, -3.1, 5.4]])
    objective_values = torch.sin(variable_values)

    variable_values = variable_values.T
    objective_values = objective_values.T

    predictor = botorch_predictor(
        problem_information={
            'n_variables': 1,
            'n_objectives': 1,
            'n_evaluations_per_step': 4,
            'bounds': bounds.tolist()
        },
        model={
            'training_settings': {
                'max_iter': 50
            }
        }
    )

    predictor.update_with_new_data(
        variable_values=variable_values,
        objective_values=objective_values,
    )

    prediction = predictor.predict_values(
        variable_values=variable_1_array
    )

    assert bool((prediction['mean'] > prediction['lower']).min()) is True
    assert bool((prediction['upper'] > prediction['mean']).min()) is True

    for prediction_band in ['mean', 'lower', 'upper']:
        assert list(prediction[prediction_band].shape) == [variable_1_array.shape[0], 1]  # type: ignore


def test_botorch_predict_values_2_objectives() -> None:
    # bounds = torch.tensor([-10.0, 10.0])
    #
    # variable_1_array = torch.arange(
    #     start=float(bounds[0]),
    #     end=float(bounds[1]),
    #     step=0.1
    # )
    #
    # variable_values = torch.tensor([
    #     [1.2, 3.2, 2.1, 5.1, -3.1, 5.4],
    #     [2.1, -2.2, -3.4, 1.2, 0.2, 0.4]
    # ])
    # objective_values = torch.vstack([
    #     torch.sin(variable_values[0]),
    #     torch.sin(variable_values[1])
    # ])
    #
    # variable_values = variable_values.T
    # objective_values = objective_values.T
    #
    # predictor = botorch_predictor(
    #     problem_information={
    #         'n_variables': 2,
    #         'n_objectives': 2,
    #         'n_evaluations_per_step': 4,
    #         'bounds': bounds.tolist()
    #     },
    #     model={
    #     'training_settings': {
    #         'max_iter': 50
    #     }
    # },
    # )
    #
    # predictor.update_with_new_data(
    #     variable_values=variable_values,
    #     objective_values=objective_values,
    # )
    #
    # _ = predictor.predict_values(
    #     variable_values=variable_1_array
    # )

    # TODO: Finish implementing test
    #   - Like the one with 1 objective above

    pass


def test_botorch_predict_values_wrong_dimensions() -> None:
    bounds = torch.tensor([-10.0, 10.0])

    variable_values = torch.tensor([
        [1.2, 3.2, 2.1, 5.1, -3.1, 5.4],
        [2.1, -2.2, -3.4, 1.2, 0.2, 0.4]
    ])
    objective_values = torch.vstack([
        torch.sin(variable_values[0]),
        torch.sin(variable_values[1])
    ])

    variable_values = variable_values.T
    objective_values = objective_values.T

    predictor = botorch_predictor(
        problem_information={
            'n_variables': 2,
            'n_objectives': 2,
            'n_evaluations_per_step': 4,
            'bounds': bounds.tolist()
        }
    )

    predictor.update_with_new_data(
        variable_values=variable_values,
        objective_values=objective_values,
    )

    # This array has the wrong dimensions
    variable_1_array = torch.arange(
        start=float(bounds[0]),
        end=float(bounds[1]),
        step=0.1
    )

    with pytest.raises(ValueError):
        _ = predictor.predict_values(
            variable_values=variable_1_array
        )


def test_botorch_predict_train_wrong_obj_dims() -> None:
    bounds = torch.tensor([-10.0, 10.0])

    variable_values = torch.tensor([
        [1.2, 3.2, 2.1, 5.1, -3.1, 5.4],
        [2.1, -2.2, -3.4, 1.2, 0.2, 0.4]
    ])
    objective_values = torch.tensor([
        [1.2, 3.1, 2.3],
        [1.4, 1.6, 3.1]
    ])

    variable_values = variable_values.T
    objective_values = objective_values.T

    predictor = botorch_predictor(
        problem_information={
            'n_variables': 2,
            'n_objectives': 2,
            'n_evaluations_per_step': 4,
            'bounds': bounds.tolist()
        }
    )

    with pytest.raises(ValueError):
        predictor.update_with_new_data(
            variable_values=variable_values,
            objective_values=objective_values,
        )


def test_botorch_update_with_new_data_fails_at_positional_args() -> None:

    bounds = torch.tensor([-10.0, 10.0])

    variable_values = torch.tensor([
        [1.2, 3.2, 2.1, 5.1, -3.1, 5.4],
        [2.1, -2.2, -3.4, 1.2, 0.2, 0.4]
    ])
    objective_values = torch.vstack([
        torch.sin(variable_values[0]),
        torch.sin(variable_values[1])
    ])

    variable_values = variable_values.T
    objective_values = objective_values.T

    predictor = botorch_predictor(
        problem_information={
            'n_variables': 2,
            'n_objectives': 2,
            'n_evaluations_per_step': 4,
            'bounds': bounds.tolist()
        }
    )

    with pytest.raises(TypeError):
        predictor.update_with_new_data(  # type: ignore
            variable_values,
            objective_values,
        )
