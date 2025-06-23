from typing import Literal, Optional, TypedDict, Union, Unpack

from veropt.optimiser.acquisition import BotorchAcquisitionFunction
from veropt.optimiser.acquisition_optimiser import AcquisitionOptimiser
from veropt.optimiser.model import AdamModelOptimiser, GPyTorchFullModel, GPyTorchSingleModel, \
    GPyTorchTrainingParametersInputDict, MaternParametersInputDict, MaternSingleModel, \
    TorchModelOptimiser

from veropt.optimiser.normalisation import Normaliser, NormaliserZeroMeanUnitVariance
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import OptimiserSettingsInputDict
from veropt.optimiser.prediction import BotorchPredictor, Predictor


# TODO: Go through naming and make consistent, good choices


SingleKernelOptions = Literal['matern']
KernelOptions = Union[SingleKernelOptions, list[SingleKernelOptions], None]
KernelOptimiserOptions = Literal['adam']
AcquisitionChoice = Union[Literal['qlehvi', 'ucb'], None]
AcquisitionOptimiserChoice = Union[Literal['dual_annealing'], None]


# TODO: In the ideal world these constructors look simple and are simple to interact with until you need
#  flexibility/complexity
#   - Look into overloaded methods


# TODO: Put defaults in json and load them when in some neat way


class BotorchPredictionChoice(TypedDict):
    model: Union[GPyTorchFullModel, 'GPytorchModelChoice']
    acquisition_function: Union[BotorchAcquisitionFunction, AcquisitionChoice]
    acquisition_optimiser: Union[AcquisitionOptimiser, AcquisitionOptimiserChoice]


def bayesian_optimiser(
        n_initial_points: int,
        n_bayesian_points: int,
        n_evaluations_per_step: int,
        objective: Union[CallableObjective, InterfaceObjective],
        predictor: Union[Predictor, BotorchPredictionChoice, None] = None,
        normaliser_class: Union[type[Normaliser], Literal['zero_mean_unit_variance'], None] = None,
        **kwargs: Unpack[OptimiserSettingsInputDict]
) -> BayesianOptimiser:

    problem_information: ProblemInformation = {
        'n_variables': objective.n_variables,
        'n_objectives': objective.n_objectives,
        'n_evaluations_per_step': n_evaluations_per_step,
        'bounds': objective.bounds.tolist(),
    }

    if isinstance(predictor, Predictor):
        built_predictor = predictor

    elif type(predictor) is dict:

        built_predictor = botorch_predictor(
            problem_information=problem_information,
            **predictor
        )

    elif predictor is None:
        built_predictor = botorch_predictor(
            problem_information=problem_information
        )

    else:
        raise ValueError("Predictor must be either a BotorchPredictionChoice dict, a Predictor object or None")

    if type(normaliser_class) is type:

            if issubclass(normaliser_class, Normaliser):
                built_normaliser_class = normaliser_class

            else:
                raise ValueError("Normaliser_class must be a subclass of Normaliser")

    else:

        built_normaliser_class = normaliser(
            normaliser_type=normaliser_class
        )

    # TODO: Implement


    return BayesianOptimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evaluations_per_step,
        objective=objective,
        predictor=built_predictor,
        normaliser_class=built_normaliser_class,
        **kwargs
    )


class ProblemInformation(TypedDict):
    n_variables: int
    n_objectives: int
    n_evaluations_per_step: int
    bounds: list[list[float]]


def botorch_predictor(
        problem_information: ProblemInformation,
        model: Union[GPyTorchFullModel, 'GPytorchModelChoice'] = None,
        acquisition_function: Union[BotorchAcquisitionFunction, AcquisitionChoice] = None,
        acquisition_optimiser: Union[AcquisitionOptimiser, AcquisitionOptimiserChoice] = None
) -> BotorchPredictor:

    if isinstance(model, GPyTorchFullModel):

        built_model = model

    else:

        built_model = gpytorch_model(
            n_variables=problem_information['n_variables'],
            n_objectives=problem_information['n_objectives'],
            **(model or {}),
        )

    if isinstance(acquisition_function, BotorchAcquisitionFunction):

        built_acquisition_function = acquisition_function

    else:
        built_acquisition_function = botorch_acquisition_function(
            aquisition_choice=acquisition_function,
            n_variables=problem_information['n_variables'],
            n_objectives=problem_information['n_objectives'],
        )

    if isinstance(acquisition_optimiser, AcquisitionOptimiser):
        built_acquisition_optimiser = acquisition_optimiser

    else:
        built_acquisition_optimiser = build_acquisition_optimiser(
            bounds=problem_information['bounds'],
            n_evaluations_per_step=problem_information['n_evaluations_per_step'],
        )


    # TODO: Figure out how to choose seq prox punish or not


    return BotorchPredictor(
        model=built_model,
        acquisition_function=built_acquisition_function,
        acquisition_optimiser=built_acquisition_optimiser,
    )


class GPytorchModelChoice(TypedDict):
    kernels: Union[KernelOptions, list[GPyTorchSingleModel]]
    kernel_optimiser: Optional[KernelOptimiserOptions]
    settings: GPyTorchTrainingParametersInputDict


def gpytorch_model(
        n_variables: int,
        n_objectives: int,
        kernels: Union[KernelOptions, list[GPyTorchSingleModel]] = None,
        kernel_optimiser: Optional[KernelOptimiserOptions] = None,
        settings: GPyTorchTrainingParametersInputDict = None,
) -> GPyTorchFullModel:

    single_model_list = gpytorch_single_model_list(
        kernels=kernels,
        n_variables=n_variables,
        n_objectives=n_objectives,
    )

    model_optimiser = torch_model_optimiser(
        kernel_optimiser=kernel_optimiser,
    )

    return GPyTorchFullModel(
        n_variables=n_variables,
        n_objectives=n_objectives,
        single_model_list=single_model_list,
        model_optimiser=model_optimiser,
        **(settings or {})
    )


def gpytorch_single_model_list(
        kernels: Union[KernelOptions, list[GPyTorchSingleModel]],
        n_variables: int,
        n_objectives: int,
) -> list[GPyTorchSingleModel]:

    wrong_kernel_input_message = (
        "'kernels' must be either None, a list of GPyTorchSingleModel, a valid kernel option or "
        "a list of valid kernel choices"
    )

    if type(kernels) is list:

        if kernels[0] == str:

            for kernel in kernels: assert type(kernel) is str, wrong_kernel_input_message

            single_model_list = []
            for kernel in kernels:
                single_model_list.append(gpytorch_single_model(
                    kernel=kernel,
                    n_variables=n_variables,
                ))

        elif isinstance(kernels[0], GPyTorchSingleModel):

            for kernel in kernels: assert isinstance(kernel, GPyTorchSingleModel), wrong_kernel_input_message

            single_model_list = kernels

        else:
            raise ValueError(wrong_kernel_input_message)

    elif type(kernels) is str:

        single_model_list = []
        for objective_no in range(n_objectives):
            single_model_list.append(gpytorch_single_model(
                kernel=kernels,
                n_variables=n_variables,
            ))

    elif kernels is None:

        single_model_list = gpytorch_single_model_list(
            kernels='matern',
            n_variables=n_variables,
            n_objectives=n_objectives,
        )

    else:
        raise ValueError(wrong_kernel_input_message)

    return single_model_list


KernelInputDict = MaternParametersInputDict  # To be expanded when more kernels are added


def _validate_typed_dict(
        dictionary: dict,
        typed_dict: type(TypedDict),
        kernel: str
) -> None:
    expected_keys = list(typed_dict.__annotations__.keys())

    for key in dictionary.keys():
        assert key in expected_keys, (
            f"Setting {key} not recognised for kernel '{kernel}'"
        )


def gpytorch_single_model(
        kernel: SingleKernelOptions,
        n_variables: int,
        settings: Optional[KernelInputDict] = None
) -> GPyTorchSingleModel:

    # TODO: Pass settings to this function higher up

    settings = settings or {}

    if kernel == 'matern':

        _validate_typed_dict(
            dictionary=settings,
            typed_dict=MaternParametersInputDict,
            kernel=kernel,
        )

        return MaternSingleModel(
            n_variables=n_variables,
            **settings
        )

    else:
        raise NotImplementedError(f"Kernel {kernel} not implemented")


def torch_model_optimiser(
        kernel_optimiser: Optional[KernelOptimiserOptions],
) -> TorchModelOptimiser:

    if kernel_optimiser == 'adam':
        return AdamModelOptimiser()

    elif kernel_optimiser is None:
        return torch_model_optimiser(kernel_optimiser='adam')

    else:
        raise NotImplementedError(f"Kernel optimiser {kernel_optimiser} not implemented")


def botorch_acquisition_function(
        aquisition_choice: AcquisitionChoice,
        n_variables: int,
        n_objectives: int,
) -> BotorchAcquisitionFunction:
    raise NotImplementedError()


def build_acquisition_optimiser(
        bounds: list[list[float]],
        n_evaluations_per_step: int
) -> AcquisitionOptimiser:

    raise NotImplementedError()

def normaliser(
        normaliser_type: Union[Literal['zero_mean_unit_variance'], None]
) -> type[Normaliser]:

    if normaliser_type == 'zero_mean_unit_variance':
        return NormaliserZeroMeanUnitVariance

    elif normaliser_type is None:
        return NormaliserZeroMeanUnitVariance

    else:
        raise ValueError(f"Unknown normaliser type: {normaliser_type}")
