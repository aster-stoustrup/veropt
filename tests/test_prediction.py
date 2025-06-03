import torch

from veropt.optimiser.acquisition import DualAnnealingOptimiser, QLogExpectedHyperVolumeImprovement, \
    UpperConfidenceBound
from veropt.optimiser.model import AdamModelOptimiser, GPyTorchFullModel, MaternSingleModel
from veropt.optimiser.prediction import BotorchPredictor


def _build_matern_model(
        n_variables: int,
        n_objectives: int,
) -> GPyTorchFullModel:

    model = GPyTorchFullModel(
        n_variables=n_variables,
        n_objectives=n_objectives,
        single_model_list=[
            MaternSingleModel(
                n_variables=n_variables,
            ) for _ in range(n_objectives)
        ],
        model_optimiser=AdamModelOptimiser()
    )

    return model


def _build_matern_predictor_ucb(
        bounds: torch.Tensor,
        n_variables: int,
        n_objectives: int,
) -> BotorchPredictor:

    model = _build_matern_model(
        n_variables=n_variables,
        n_objectives=n_objectives
    )

    acquisition_function = UpperConfidenceBound()

    acquisition_optimiser = DualAnnealingOptimiser(
        bounds=bounds
    )

    predictor = BotorchPredictor(
        model=model,
        acquisition_function=acquisition_function,
        acquisition_optimiser=acquisition_optimiser,
    )

    return predictor


def _build_matern_predictor_qlogehvi(
        bounds: torch.Tensor,
        n_variables: int,
        n_objectives: int,
) -> BotorchPredictor:

    model = _build_matern_model(
        n_variables=n_variables,
        n_objectives=n_objectives
    )

    acquisition_function = QLogExpectedHyperVolumeImprovement()

    acquisition_optimiser = DualAnnealingOptimiser(
        bounds=bounds
    )

    predictor = BotorchPredictor(
        model=model,
        acquisition_function=acquisition_function,
        acquisition_optimiser=acquisition_optimiser,
    )

    return predictor


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

    predictor = _build_matern_predictor_ucb(
        bounds=bounds,
        n_variables=1,
        n_objectives=1,
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
        assert list(prediction[prediction_band].shape) == [variable_1_array.shape[0], 1]


def test_botorch_predict_values_2_objectives() -> None:

    bounds = torch.tensor([-10.0, 10.0])

    variable_1_array = torch.arange(
        start=float(bounds[0]),
        end=float(bounds[1]),
        step=0.1
    )

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

    predictor = _build_matern_predictor_qlogehvi(
        bounds=bounds,
        n_variables=2,
        n_objectives=2,
    )

    predictor.update_with_new_data(
        variable_values=variable_values,
        objective_values=objective_values,
    )

    prediction = predictor.predict_values(
        variable_values=variable_1_array
    )

    assert False
