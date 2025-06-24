from typing import Any, Literal, Mapping, Optional, TypedDict, Union, Unpack, get_args

import torch

from veropt.optimiser.acquisition import BotorchAcquisitionFunction, QLogExpectedHyperVolumeImprovement, \
    UpperConfidenceBound, UpperConfidenceBoundOptions
from veropt.optimiser.acquisition_optimiser import AcquisitionOptimiser, DualAnnealingOptimiser, DualAnnealingSettings, \
    ProximityPunishSettings, ProximityPunishmentSequentialOptimiser
from veropt.optimiser.model import AdamModelOptimiser, GPyTorchFullModel, GPyTorchSingleModel, \
    GPyTorchTrainingParametersInputDict, MaternParametersInputDict, MaternSingleModel, \
    TorchModelOptimiser
from veropt.optimiser.normalisation import Normaliser, NormaliserZeroMeanUnitVariance
from veropt.optimiser.objective import CallableObjective, InterfaceObjective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import OptimiserSettingsInputDict
from veropt.optimiser.prediction import BotorchPredictor


SingleKernelOptions = Literal['matern']
KernelOptions = Union[SingleKernelOptions, list[SingleKernelOptions]]
KernelOptimiserOptions = Literal['adam']
AcquisitionOptions = Literal['qlogehvi', 'ucb']
AcquisitionOptimiserOptions = Literal['dual_annealing']
NormaliserChoice = Literal['zero_mean_unit_variance']

# TODO: Go through naming and make consistent, good choices

# TODO: Consider making a function that can give valid arguments to the user?
#   - Like something that prints out an overview of options
#   - Should probably live in some documentation somewhere actually...?

# TODO: Put defaults in json and load them when in some neat way
#   - Maybe make a test to ensure file is found and loaded across installations/platforms


def bayesian_optimiser(
        n_initial_points: int,
        n_bayesian_points: int,
        n_evaluations_per_step: int,
        objective: Union[CallableObjective, InterfaceObjective],
        model: Union[GPyTorchFullModel, 'GPytorchModelChoice', None] = None,
        acquisition_function: Union[BotorchAcquisitionFunction, 'AcquisitionChoice', None] = None,
        acquisition_optimiser: Union[AcquisitionOptimiser, 'AcquisitionOptimiserChoice', None] = None,
        normaliser: Union[type[Normaliser], NormaliserChoice, None] = None,
        **kwargs: Unpack[OptimiserSettingsInputDict]
) -> BayesianOptimiser:

    problem_information: ProblemInformation = {
        'n_variables': objective.n_variables,
        'n_objectives': objective.n_objectives,
        'n_evaluations_per_step': n_evaluations_per_step,
        'bounds': objective.bounds.tolist(),
    }

    built_predictor = botorch_predictor(
        problem_information=problem_information,
        model=model,
        acquisition_function=acquisition_function,
        acquisition_optimiser=acquisition_optimiser,
    )

    if type(normaliser) is type:

        if issubclass(normaliser, Normaliser):
            normaliser_class = normaliser

        else:
            raise ValueError("Normaliser_class must be a subclass of Normaliser")

    else:

        normaliser_class = build_normaliser(
            normaliser_choice=normaliser  # type: ignore[arg-type]  # checked above with 'issubclass'
        )

    return BayesianOptimiser(
        n_initial_points=n_initial_points,
        n_bayesian_points=n_bayesian_points,
        n_evaluations_per_step=n_evaluations_per_step,
        objective=objective,
        predictor=built_predictor,
        normaliser_class=normaliser_class,
        **kwargs
    )


class ProblemInformation(TypedDict):
    n_variables: int
    n_objectives: int
    n_evaluations_per_step: int
    bounds: list[list[float]]


def botorch_predictor(
        problem_information: ProblemInformation,
        model: Optional[Union[GPyTorchFullModel, 'GPytorchModelChoice']] = None,
        acquisition_function: Union[BotorchAcquisitionFunction, 'AcquisitionChoice', None] = None,
        acquisition_optimiser: Union[AcquisitionOptimiser, 'AcquisitionOptimiserChoice', None] = None
) -> BotorchPredictor:

    if isinstance(model, GPyTorchFullModel):

        built_model = model

    else:

        built_model = gpytorch_model(
            n_variables=problem_information['n_variables'],
            n_objectives=problem_information['n_objectives'],
            **model or {},
        )

    if isinstance(acquisition_function, BotorchAcquisitionFunction):

        built_acquisition_function = acquisition_function

    else:
        built_acquisition_function = botorch_acquisition_function(
            n_variables=problem_information['n_variables'],
            n_objectives=problem_information['n_objectives'],
            **acquisition_function or {}
        )

    if isinstance(acquisition_optimiser, AcquisitionOptimiser):
        built_acquisition_optimiser = acquisition_optimiser

    else:
        built_acquisition_optimiser = build_acquisition_optimiser(
            bounds=problem_information['bounds'],
            n_evaluations_per_step=problem_information['n_evaluations_per_step'],
            **acquisition_optimiser or {}
        )

    return BotorchPredictor(
        model=built_model,
        acquisition_function=built_acquisition_function,
        acquisition_optimiser=built_acquisition_optimiser,
    )


class GPytorchModelChoice(TypedDict, total=False):
    kernels: Union[KernelOptions, list[GPyTorchSingleModel], None]
    kernel_settings: Optional['KernelInputDict']
    kernel_optimiser: Optional[KernelOptimiserOptions]
    training_settings: Optional[GPyTorchTrainingParametersInputDict]


def gpytorch_model(
        n_variables: int,
        n_objectives: int,
        kernels: Union[KernelOptions, list[GPyTorchSingleModel], None] = None,
        kernel_settings: Union['KernelInputDict', list['KernelInputDict'], None] = None,
        kernel_optimiser: Optional[KernelOptimiserOptions] = None,
        training_settings: Optional[GPyTorchTrainingParametersInputDict] = None,
) -> GPyTorchFullModel:

    single_model_list = gpytorch_single_model_list(
        n_variables=n_variables,
        n_objectives=n_objectives,
        kernels=kernels,
        kernel_settings=kernel_settings
    )

    model_optimiser = torch_model_optimiser(
        kernel_optimiser=kernel_optimiser,
    )

    return GPyTorchFullModel(
        n_variables=n_variables,
        n_objectives=n_objectives,
        single_model_list=single_model_list,
        model_optimiser=model_optimiser,
        **(training_settings or {})
    )


def gpytorch_single_model_list(
        n_variables: int,
        n_objectives: int,
        kernels: Union[KernelOptions, list[GPyTorchSingleModel], None] = None,
        kernel_settings: Union['KernelInputDict', list['KernelInputDict'], None] = None
) -> list[GPyTorchSingleModel]:

    wrong_kernel_input_message = (
        "'kernels' must be either None, a list of GPyTorchSingleModel, a valid kernel option or "
        "a list of valid kernel choices"
    )

    if type(kernels) is list:

        assert len(kernels) == n_objectives, (
            f"Please specify a kernel choice for each objective. "
            f"Received {n_objectives} objectives but {len(kernels)} kernels."
        )

        if kernels[0] == str:

            for kernel in kernels:
                assert type(kernel) is str, wrong_kernel_input_message

            if kernel_settings is not None:
                assert type(kernel_settings) is list, (
                    "'kernel_settings' must be a list of dicts if 'kernels' is a list of strings."
                )
                assert len(kernel_settings) == n_objectives

            else:
                kernel_settings = [{}] * n_objectives

            single_model_list = []
            for kernel_no, kernel in enumerate(kernels):  # type: ignore[assignment]  # checked above, it's 'list[str]'
                single_model_list.append(gpytorch_single_model(
                    n_variables=n_variables,
                    kernel=kernel,  # type: ignore[arg-type]  # checked above, kernel is 'str'
                    settings=kernel_settings[kernel_no]
                ))

        elif isinstance(kernels[0], GPyTorchSingleModel):

            assert kernel_settings is None, "Cannot accept kernel settings for an already created model list."

            for kernel in kernels:
                assert isinstance(kernel, GPyTorchSingleModel), wrong_kernel_input_message

            single_model_list = kernels  # type: ignore[assignment]  # (type is checked above, mypy can't follow it)

        else:
            raise ValueError(wrong_kernel_input_message)

    elif type(kernels) is str:

        if kernel_settings is not None:
            assert type(kernel_settings) is dict, (
                "'kernel_settings' must be None or a single dict if 'kernels' is a single string."
            )

        single_model_list = []
        for objective_no in range(n_objectives):
            single_model_list.append(gpytorch_single_model(
                n_variables=n_variables,
                kernel=kernels,
                settings=kernel_settings
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


KernelInputDict = Union[MaternParametersInputDict]  # To be expanded when more kernels are added
TypedDictHint = Mapping[str, Any]  # type:ignore[explicit-any]


def _validate_typed_dict(
        typed_dict: TypedDictHint,
        expected_typed_dict_class: type,
        object_name: str
) -> None:
    expected_keys = list(expected_typed_dict_class.__annotations__.keys())

    for key in typed_dict.keys():
        assert key in expected_keys, (
            f"Option '{key}' not recognised for '{object_name}'. Expected options: {expected_keys}."
        )


def gpytorch_single_model(
        n_variables: int,
        kernel: Optional[SingleKernelOptions] = None,
        settings: Optional[KernelInputDict] = None
) -> GPyTorchSingleModel:

    settings = settings or {}

    if kernel == 'matern':

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=MaternParametersInputDict,
            object_name=kernel,
        )

        return MaternSingleModel(
            n_variables=n_variables,
            **settings
        )

    elif kernel is None:

        return gpytorch_single_model(
            n_variables=n_variables,
            kernel='matern',
            settings=settings
        )

    else:
        raise NotImplementedError(
            f"Kernel '{kernel}' not recognised. Implemented kernels are: {get_args(KernelOptions)}"
        )


def torch_model_optimiser(
        kernel_optimiser: Optional[KernelOptimiserOptions],
) -> TorchModelOptimiser:

    if kernel_optimiser == 'adam':
        return AdamModelOptimiser()

    elif kernel_optimiser is None:
        return torch_model_optimiser(kernel_optimiser='adam')

    else:
        raise NotImplementedError(f"Kernel optimiser {kernel_optimiser} not implemented")


AcquisitionSettings = UpperConfidenceBoundOptions  # expand with more options when adding acq_funcs


class AcquisitionChoice(TypedDict, total=False):
    function: Optional[AcquisitionOptions]
    parameters: Optional[AcquisitionSettings]


def botorch_acquisition_function(
        n_variables: int,
        n_objectives: int,
        function: Optional[AcquisitionOptions] = None,
        parameters: Optional[AcquisitionSettings] = None
) -> BotorchAcquisitionFunction:

    if function  is None:
        raise NotImplementedError()

    elif function == 'qlogehvi':

        assert parameters is None or parameters == {}, "'qlogehvi' does not take any parameters."

        return QLogExpectedHyperVolumeImprovement(
            n_variables=n_variables,
            n_objectives=n_objectives
        )
        
    elif function == 'ucb':

        if parameters is not None:

            _validate_typed_dict(
                typed_dict=parameters,
                expected_typed_dict_class=AcquisitionSettings,
                object_name=function
            )

        return UpperConfidenceBound(
            n_variables=n_variables,
            n_objectives=n_objectives,
            **parameters or {}
        )

    else:
        raise ValueError(f"acquisition_choice must be None or {get_args(AcquisitionChoice)}")


AcquisitionOptimiserSettings = DualAnnealingSettings  # expand when adding more options


class AcquisitionOptimiserChoice(TypedDict, total=False):
    optimiser: Optional[AcquisitionOptimiserOptions]
    optimiser_settings: Optional[AcquisitionOptimiserSettings]
    allow_proximity_punish: bool
    proximity_punish_settings: Optional[ProximityPunishSettings]


def build_acquisition_optimiser(
        bounds: list[list[float]],
        n_evaluations_per_step: int,
        optimiser: Optional[AcquisitionOptimiserOptions] = None,
        optimiser_settings: Optional[AcquisitionOptimiserSettings] = None,
        allow_proximity_punish: bool = True,
        proximity_punish_settings: Optional[ProximityPunishSettings] = None
) -> AcquisitionOptimiser:

    # TODO: Need to build a more general version of this where we grab the acq opt class and check allows
    #  n_evals_per_step
    #   - Still need a specific place to build the single step opt
    #     but then let it be eaten by prox punish if needed+allowed
    #   - Probably need an extra function to make this nice?

    if optimiser is None:

        raise NotImplementedError("coming asap")

    elif optimiser == 'dual_annealing':

        if n_evaluations_per_step == 1:

            if optimiser_settings is not None:
                _validate_typed_dict(
                    typed_dict=optimiser_settings,
                    expected_typed_dict_class=DualAnnealingSettings,
                    object_name=optimiser
                )

            return DualAnnealingOptimiser(
                bounds=torch.tensor(bounds),
                n_evaluations_per_step=n_evaluations_per_step,
                **optimiser_settings or {}
            )

        elif n_evaluations_per_step > 1 and allow_proximity_punish:

            if optimiser_settings is not None:
                _validate_typed_dict(
                    typed_dict=optimiser_settings,
                    expected_typed_dict_class=DualAnnealingSettings,
                    object_name=optimiser
                )

            single_step_optimiser = DualAnnealingOptimiser(
                bounds=torch.tensor(bounds),
                n_evaluations_per_step=1,
                **optimiser_settings or {}
            )

            if proximity_punish_settings is not None:
                _validate_typed_dict(
                    typed_dict=proximity_punish_settings,
                    expected_typed_dict_class=ProximityPunishSettings,
                    object_name='proximity_punish'
                )

            return ProximityPunishmentSequentialOptimiser(
                bounds=torch.tensor(bounds),
                n_evaluations_per_step=n_evaluations_per_step,
                single_step_optimiser=single_step_optimiser,
                **proximity_punish_settings or {}
            )

        elif n_evaluations_per_step > 1 and allow_proximity_punish is False:

            raise ValueError(
                "Acquisition Optimiser 'Dual Annealing' can only find one point per step."
                "Either allow using proximity punish or choose a different acquisition function optimiser."
            )

        else:
            raise RuntimeError()  # Is it possible to end up here? Probably not. GLHF if you did :))


def build_normaliser(
        normaliser_choice: Union[Literal['zero_mean_unit_variance'], None]
) -> type[Normaliser]:

    if normaliser_choice == 'zero_mean_unit_variance':
        return NormaliserZeroMeanUnitVariance

    elif normaliser_choice is None:
        return NormaliserZeroMeanUnitVariance

    else:
        raise ValueError(f"Unknown normaliser type: {normaliser_choice}")
