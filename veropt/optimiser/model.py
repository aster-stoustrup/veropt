import abc
import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Mapping, Optional, Self, Sequence, TypedDict, Union, Unpack

import botorch
import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval, LessThan
from gpytorch.distributions import MultivariateNormal

from veropt.optimiser.saver_loader_utility import SavableClass, SavableDataClass, rehydrate_object
from veropt.optimiser.utility import _validate_typed_dict, \
    check_variable_and_objective_shapes, \
    check_variable_objective_values_matching, \
    enforce_amount_of_positional_arguments, unpack_variables_objectives_from_kwargs


# TODO: Consider deleting this abstraction. Does it have a function at this point?
class SurrogateModel(metaclass=abc.ABCMeta):

    def __init__(
            self,
            n_variables: int,
            n_objectives: int
    ):
        self.n_variables = n_variables
        self.n_objectives = n_objectives

    @abc.abstractmethod
    def train_model(
            self,
            *,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:
        pass


class GPyTorchDataModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):  # type: ignore[misc]
    _num_outputs = 1

    def __init__(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            mean_module: gpytorch.means.Mean,
            kernel: gpytorch.kernels.Kernel
    ) -> None:

        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood
        )

        self.mean_module = mean_module
        self.covar_module = kernel

        self.to(tensor=train_inputs)  # making sure we're on the right device/dtype

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# TODO: Move to different file
def format_json_state_dict(
        state_dict: dict,
) -> dict:
    formatted_dict = state_dict.copy()

    for key, value in state_dict.items():

        if isinstance(value, str):

            if "inf" in value:
                formatted_dict[key] = torch.tensor(float(value))

        elif isinstance(value, list):

            for item_no, item in enumerate(value):
                if isinstance(item, str):
                    if "inf" in item:
                        value[item_no] = float(item)

            formatted_dict[key] = torch.tensor(value)

        elif isinstance(value, float):

            formatted_dict[key] = torch.tensor(value)

    return formatted_dict


class GPyTorchSingleModel(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            mean_module: gpytorch.means.Mean,
            kernel: gpytorch.kernels.Kernel,
            n_variables: int,
    ) -> None:

        self.likelihood = likelihood
        self.mean_module = mean_module
        self.kernel = kernel

        self.n_variables = n_variables

        self.model_with_data: Optional[GPyTorchDataModel] = None

        self.trained_parameters: list[dict[str, Iterator[torch.nn.Parameter]]] = [{}]

        assert 'name' in self.__class__.__dict__, (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

        assert 'name' != 'meta', (
            f"Must give subclass '{self.__class__.__name__}' the static class variable 'name'."
        )

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"trained: {'yes' if self.model_with_data else 'no'}"
            f", settings: {self.get_settings()}"
            f")"
        )

    @classmethod
    @abc.abstractmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any]
    ) -> Self:
        pass

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        model = cls.from_n_variables_and_settings(
            n_variables=saved_state['n_variables'],
            settings=saved_state['settings']
        )

        if len(saved_state['state_dict']) > 0:
            model.initialise_model_from_state_dict(
                train_inputs=torch.tensor(saved_state['train_inputs']),
                train_targets=torch.tensor(saved_state['train_targets']),
                state_dict=format_json_state_dict(saved_state['state_dict']),
            )

        return model

    @abc.abstractmethod
    def get_settings(self) -> SavableDataClass:
        pass

    @abc.abstractmethod
    def _set_up_trained_parameters(self) -> None:
        pass

    @abc.abstractmethod
    def _set_up_model_constraints(self) -> None:
        pass

    def initialise_model_with_data(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
    ) -> None:

        self.model_with_data = GPyTorchDataModel(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=self.likelihood,
            mean_module=self.mean_module,
            kernel=self.kernel
        )

        self._set_up_trained_parameters()

        self._set_up_model_constraints()

    def initialise_model_from_state_dict(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
            state_dict: dict
    ) -> None:

        self.initialise_model_with_data(
            train_inputs=train_inputs,
            train_targets=train_targets
        )

        self.model_with_data.load_state_dict(  # type: ignore[union-attr]  # model initialised just before calling this
            state_dict=state_dict
        )

    def gather_dicts_to_save(self) -> dict:

        if self.model_with_data is not None:
            state_dict = self.model_with_data.state_dict()
            train_inputs = self.model_with_data.train_inputs
            train_targets = self.model_with_data.train_targets

        else:
            state_dict = {}
            train_inputs = None
            train_targets = None

        return {
            'name': self.name,
            'state': {
                'state_dict': state_dict,
                'train_inputs': train_inputs,
                'train_targets': train_targets,
                'n_variables': self.n_variables,
                'settings': self.get_settings().gather_dicts_to_save()
            }
        }

    def set_constraint(
            self,
            constraint: Union[Interval, GreaterThan, LessThan],
            parameter_name: str,
            module: str,
            second_module: Optional[str] = None
    ) -> None:
        if self.model_with_data is not None:

            if second_module is None:

                self.model_with_data.__getattr__(module).register_constraint(
                    param_name=parameter_name,
                    constraint=constraint
                )

            else:

                self.model_with_data.__getattr__(module).__getattr__(second_module).register_constraint(
                    param_name=parameter_name,
                    constraint=constraint
                )

        else:
            # Might want to store these constraints and feed them to the trained model when it's made?
            raise NotImplementedError("Currently don't support setting constraints before model is given data.")

    def change_interval_constraints(
            self,
            lower_bound: float,
            upper_bound: float,
            parameter_name: str,
            module: str,
            second_module: Optional[str] = None
    ) -> None:
        constraint = Interval(
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        self.set_constraint(
            constraint=constraint,
            parameter_name=parameter_name,
            module=module,
            second_module=second_module
        )

    def change_greater_than_constraint(
            self,
            lower_bound: float,
            parameter_name: str,
            module: str,
            second_module: Optional[str] = None

    ) -> None:
        constraint = GreaterThan(
            lower_bound=lower_bound
        )

        self.set_constraint(
            constraint=constraint,
            parameter_name=parameter_name,
            module=module,
            second_module=second_module
        )

    def change_less_than_constraint(
            self,
            upper_bound: float,
            parameter_name: str,
            module: str,
            second_module: Optional[str] = None
    ) -> None:
        constraint = LessThan(
            upper_bound=upper_bound
        )

        self.set_constraint(
            constraint=constraint,
            parameter_name=parameter_name,
            module=module,
            second_module=second_module
        )


class MaternParametersInputDict(TypedDict, total=False):
    lengthscale_lower_bound: float
    lengthscale_upper_bound: float
    noise: float
    noise_lower_bound: float
    train_noise: bool


@dataclass
class MaternParameters(SavableDataClass):
    lengthscale_lower_bound: float = 0.1
    lengthscale_upper_bound: float = 2.0
    noise: float = 1e-8
    noise_lower_bound: float = 1e-8
    train_noise: bool = False


class DoubleMaternParametersInputDict(TypedDict, total=False):
    lengthscale_long_lower_bound: float
    lengthscale_long_upper_bound: float
    lengthscale_short_lower_bound: float
    lengthscale_short_upper_bound: float
    noise: float
    noise_lower_bound: float
    train_noise: bool


@dataclass
class DoubleMaternParameters(SavableDataClass):
    lengthscale_long_lower_bound: float = 0.1
    lengthscale_long_upper_bound: float = 2.0
    lengthscale_short_lower_bound: float = 0.001
    lengthscale_short_upper_bound: float = 0.1
    noise: float = 1e-8
    noise_lower_bound: float = 1e-8
    train_noise: bool = False


class MaternSingleModel(GPyTorchSingleModel):

    name = 'matern'

    def __init__(
            self,
            n_variables: int,
            **settings: Unpack[MaternParametersInputDict]
    ):

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        kernel = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([])
        )

        self.settings = MaternParameters(
            **settings
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any]
    ) -> 'MaternSingleModel':

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=MaternParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            **settings
        )

    def _set_up_trained_parameters(self) -> None:

        parameter_group_list = []

        assert self.model_with_data is not None, "Model must be initialised to use this function."

        if self.settings.train_noise:

            parameter_group_list.append(
                {'params': self.model_with_data.parameters()}
            )

        else:

            parameter_group_list.append(
                {'params': self.model_with_data.mean_module.parameters()}
            )

            parameter_group_list.append(
                {'params': self.model_with_data.covar_module.parameters()}
            )

        self.trained_parameters = parameter_group_list

    def _set_up_model_constraints(self) -> None:

        self.change_lengthscale_constraints(
            lower_bound=self.settings.lengthscale_lower_bound,
            upper_bound=self.settings.lengthscale_upper_bound
        )

        self.set_noise(
            noise=self.settings.noise
        )

        self.set_noise_constraint(
            lower_bound=self.settings.noise_lower_bound
        )

    def change_lengthscale_constraints(
            self,
            lower_bound: float,
            upper_bound: float
    ) -> None:

        self.change_interval_constraints(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            module='covar_module',
            parameter_name='raw_lengthscale'
        )

    def get_lengthscale(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return self.model_with_data.covar_module.lengthscale

    def set_noise(
            self,
            noise: float
    ) -> None:

        if self.model_with_data is not None:

            if noise < self.likelihood.noise_covar.raw_noise_constraint.lower_bound:
                noise = self.likelihood.noise_covar.raw_noise_constraint.lower_bound

            self.model_with_data.likelihood.noise = torch.tensor(float(noise))

        else:
            raise NotImplementedError("Currently don't support setting constraints before model is given data.")

    def set_noise_constraint(
            self,
            lower_bound: float
    ) -> None:

        # Default seems to be 1e-4
        #   - Would like to make sure we don't have noise when we try to set it to zero
        #   - Alternatively, setting it too low might risk numerical instability?

        assert self.model_with_data is not None, "Model must be initiated to change constraints"

        self.change_greater_than_constraint(
            lower_bound=lower_bound,
            parameter_name='raw_noise',
            module='likelihood',
            second_module='noise_covar'
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings


# TODO: Make prettier, this is a midnight draft :))
#   - Might be more correct to just use spectralmixturekernel than this
#       - Remember the diff between RBF and matern (SMK seems to be added RBF's)
class DoubleMaternKernelSingleModel(GPyTorchSingleModel):

    name = 'double_matern'

    def __init__(
            self,
            n_variables: int,
            **settings: Unpack[DoubleMaternParametersInputDict]
    ):

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()

        kernel_0 = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([])
        )

        kernel_1 = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([])
        )

        kernel = kernel_0 + kernel_1

        self.settings = DoubleMaternParameters(
            **settings
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel,
            n_variables=n_variables
        )

    @classmethod
    def from_n_variables_and_settings(
            cls,
            n_variables: int,
            settings: Mapping[str, Any]
    ) -> 'DoubleMaternKernelSingleModel':

        _validate_typed_dict(
            typed_dict=settings,
            expected_typed_dict_class=DoubleMaternParametersInputDict,
            object_name=cls.name,
        )

        return cls(
            n_variables=n_variables,
            **settings
        )

    def _set_up_trained_parameters(self) -> None:

        parameter_group_list = []

        assert self.model_with_data is not None, "Model must be initialised to use this function."

        if self.settings.train_noise:

            parameter_group_list.append(
                {'params': self.model_with_data.parameters()}
            )

        else:

            parameter_group_list.append(
                {'params': self.model_with_data.mean_module.parameters()}
            )

            parameter_group_list.append(
                {'params': self.model_with_data.covar_module.parameters()}
            )

        self.trained_parameters = parameter_group_list

    def _set_up_model_constraints(self) -> None:

        self.change_lengthscale_constraints(
            kernel_number=0,
            lower_bound=self.settings.lengthscale_long_lower_bound,
            upper_bound=self.settings.lengthscale_long_upper_bound
        )

        self.change_lengthscale_constraints(
            kernel_number=1,
            lower_bound=self.settings.lengthscale_short_lower_bound,
            upper_bound=self.settings.lengthscale_short_upper_bound
        )

        self.set_noise(
            noise=self.settings.noise
        )

        self.set_noise_constraint(
            lower_bound=self.settings.noise_lower_bound
        )

    def change_lengthscale_constraints(
            self,
            kernel_number: int,
            lower_bound: float,
            upper_bound: float
    ) -> None:

        assert self.model_with_data is not None, "Model must be initialised to use this method"

        # TODO: Ideally use normal system
        #   - Might need to make that system more general

        constraint = Interval(
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )

        self.model_with_data.covar_module.kernels[kernel_number].register_constraint(
            param_name='raw_lengthscale',
            constraint=constraint
        )

    def get_lengthscale(self) -> torch.Tensor:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        raise NotImplementedError()

    def set_noise(
            self,
            noise: float
    ) -> None:

        if self.model_with_data is not None:

            if noise < self.likelihood.noise_covar.raw_noise_constraint.lower_bound:
                noise = self.likelihood.noise_covar.raw_noise_constraint.lower_bound

            self.model_with_data.likelihood.noise = torch.tensor(float(noise))

        else:
            raise NotImplementedError("Currently don't support setting constraints before model is given data.")

    def set_noise_constraint(
            self,
            lower_bound: float
    ) -> None:

        # Default seems to be 1e-4
        #   - Would like to make sure we don't have noise when we try to set it to zero
        #   - Alternatively, setting it too low might risk numerical instability?

        assert self.model_with_data is not None, "Model must be initiated to change constraints"

        self.change_greater_than_constraint(
            lower_bound=lower_bound,
            parameter_name='raw_noise',
            module='likelihood',
            second_module='noise_covar'
        )

    def get_settings(self) -> SavableDataClass:
        return self.settings


class TorchModelOptimiser(SavableClass, metaclass=abc.ABCMeta):

    name: str = 'meta'

    def __init__(
            self,
            optimiser_class: type[torch.optim.Optimizer],
            optimiser_settings: Optional[dict] = None
    ) -> None:

        self.optimiser: Optional[torch.optim.Optimizer] = None
        self.optimiser_class = optimiser_class

        # TODO: Should probably do the dataclass thing here as well
        self.optimiser_settings = optimiser_settings or {}

    def gather_dicts_to_save(self) -> dict:

        return {
            'name': self.name,
            'state': {
                'settings': self.optimiser_settings
            }
        }

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        # TODO: Make this nicer?
        #   - This is essentially assuming an init like in the Adam implementation
        return cls(
            **saved_state['settings']
        )

    def initiate_optimiser(
            self,
            parameters: Iterator[torch.nn.Parameter] | list[dict[str, Iterator[torch.nn.Parameter]]]
    ) -> None:

        self.optimiser = self.optimiser_class(
            params=parameters,
            **self.optimiser_settings
        )


class AdamModelOptimiser(TorchModelOptimiser):

    name = 'adam'

    def __init__(
            self,
            **kwargs: dict
    ) -> None:

        for key in kwargs.keys():
            assert key in torch.optim.Adam.__init__.__code__.co_varnames, (
                f"{key} is not an accepted argument for the torch optimiser 'Adam'."
            )

        super().__init__(
            optimiser_class=torch.optim.Adam,
            optimiser_settings=kwargs
        )


class ModelMode(Enum):
    training = 1
    evaluating = 2


class GPyTorchTrainingParametersInputDict(TypedDict, total=False):
    learning_rate: float
    loss_change_to_stop: float
    max_iter: int
    verbose: bool


@dataclass
class GPyTorchTrainingParameters(SavableDataClass):
    learning_rate: float = 0.1
    loss_change_to_stop: float = 1e-6  # TODO: Find optimal value for this?
    max_iter: int = 10_000
    verbose: bool = True


class GPyTorchFullModel(SurrogateModel, SavableClass):

    name = 'gpytorch_full_model'

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
            single_model_list: Sequence[GPyTorchSingleModel],
            model_optimiser: TorchModelOptimiser,
            training_settings: GPyTorchTrainingParameters
    ) -> None:

        self.training_settings = training_settings

        assert len(single_model_list) == n_objectives, "Number of objectives must match the length of the model list"

        self._model_list = single_model_list

        self._model: Optional[botorch.models.ModelListGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.LikelihoodList] = None

        self._marginal_log_likelihood: Optional[gpytorch.mlls.SumMarginalLogLikelihood] = None

        self._model_optimiser = model_optimiser

        if self._model_list[0].model_with_data is None:
            single_models_are_trained = False

        else:
            single_models_are_trained = True

            for model in self._model_list:
                assert model.model_with_data is not None

        if single_models_are_trained:
            self._initialise_model_likelihood_lists()

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    @classmethod
    def from_the_beginning(
            cls,
            n_variables: int,
            n_objectives: int,
            single_model_list: Sequence[GPyTorchSingleModel],
            model_optimiser: TorchModelOptimiser,
            **kwargs: Unpack[GPyTorchTrainingParametersInputDict]
    ) -> 'GPyTorchFullModel':

        training_settings = GPyTorchTrainingParameters(
            **kwargs
        )

        return cls(
            n_variables=n_variables,
            n_objectives=n_objectives,
            single_model_list=single_model_list,
            model_optimiser=model_optimiser,
            training_settings=training_settings
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        model_list = []

        for model_dict in saved_state['model_dicts'].values():
            model_list.append(rehydrate_object(
                superclass=GPyTorchSingleModel,
                name=model_dict['name'],
                saved_state=model_dict['state']
            ))

        assert len(model_list) == saved_state['n_objectives']

        settings = GPyTorchTrainingParameters.from_saved_state(
            saved_state=saved_state['settings']
        )

        model_optimiser = rehydrate_object(
            superclass=TorchModelOptimiser,
            name=saved_state['model_optimiser']['name'],
            saved_state=saved_state['model_optimiser']['state']
        )

        return cls(
            n_variables=saved_state['n_variables'],
            n_objectives=saved_state['n_objectives'],
            single_model_list=model_list,
            model_optimiser=model_optimiser,
            training_settings=settings
        )

    @staticmethod
    def _check_input_dimensions[T, **P](
            function: Callable[P, T]
    ) -> Callable[P, T]:

        @functools.wraps(function)
        def check_dimensions(
                *args: P.args,
                **kwargs: P.kwargs,
        ) -> T:

            enforce_amount_of_positional_arguments(
                function=function,
                received_args=args
            )

            assert isinstance(args[0], GPyTorchFullModel)
            self: GPyTorchFullModel = args[0]

            variable_values, objective_values = unpack_variables_objectives_from_kwargs(kwargs)

            if variable_values is None and objective_values is None:
                raise RuntimeError("This decorator was called to check input shapes but found no valid inputs.")

            check_variable_and_objective_shapes(
                n_variables=self.n_variables,
                n_objectives=self.n_objectives,
                function_name=function.__name__,
                class_name=self.__class__.__name__,
                variable_values=variable_values,
                objective_values=objective_values,
            )

            return function(
                *args,
                **kwargs
            )

        return check_dimensions

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}({[model.__class__.__name__ for model in self._model_list]})"
        )

    def __str__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"\n{''.join([f"{model} \n" for model in self._model_list])}"
            f"settings: {self.training_settings}\n"
            f")"
        )

    def __getitem__(self, model_no: int) -> GPyTorchSingleModel:

        if model_no > len(self._model_list) - 1:
            raise IndexError()

        return self._model_list[model_no]

    def __len__(self) -> int:
        return len(self._model_list)

    @_check_input_dimensions
    def __call__(
            self,
            *,
            variable_values: torch.Tensor
    ) -> list[MultivariateNormal]:

        previous_mode = self._mode

        self._set_mode_evaluate()

        assert self._likelihood is not None, "Model must be initiated to call it"
        assert self._model is not None, "Model must be initiated to call it"

        estimated_objective_values = self._likelihood(
            *self._model(
                *([variable_values] * self.n_objectives)
            )
        )

        self._set_mode(model_mode=previous_mode)

        return estimated_objective_values

    def gather_dicts_to_save(self) -> dict:

        model_dicts: dict[str, dict] = {}

        for model_no, model in enumerate(self._model_list):
            model_dicts[f'model_{model_no}'] = model.gather_dicts_to_save()

        return {
            'name': self.name,
            'state': {
                'model_dicts': model_dicts,
                'settings': self.training_settings.gather_dicts_to_save(),
                'model_optimiser': self._model_optimiser.gather_dicts_to_save(),
                'n_variables': self.n_variables,
                'n_objectives': self.n_objectives,
            }
        }

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def train_model(
            self,
            *,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:

        self.initialise_model(
            variable_values=variable_values,
            objective_values=objective_values
        )

        self._set_mode_train()

        self._marginal_log_likelihood = gpytorch.mlls.SumMarginalLogLikelihood(
            likelihood=self._likelihood,
            model=self._model
        )

        self._initiate_optimiser()

        self._train_backwards()

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def initialise_model(
            self,
            *,
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:

        for objective_number in range(self.n_objectives):

            self._model_list[objective_number].initialise_model_with_data(
                train_inputs=variable_values,
                train_targets=objective_values[:, objective_number]
            )

        self._initialise_model_likelihood_lists()

    def _initialise_model_likelihood_lists(self) -> None:

        for model in self._model_list:
            assert model.model_with_data is not None, "Single models must be trained to use this function"
            assert model.model_with_data.likelihood is not None, "Single models must be trained to use this function"

        # TODO: Might need to look into more options here
        #   - Currently seems to be assuming independent models. Maybe need to add an option for this?
        #       - Probably would need a different class (GPyTorchIndependentModels vs the opposite)
        #       - Botorch has some options for this
        self._model = botorch.models.ModelListGP(
            *[model.model_with_data for model in self._model_list]  # type: ignore  # (type is checked above)
        )
        self._likelihood = gpytorch.likelihoods.LikelihoodList(
            *[model.model_with_data.likelihood for model in self._model_list]  # type: ignore[union-attr]
        )

    def get_gpytorch_model(self) -> botorch.models.ModelListGP:

        assert self._model is not None, "Model must be initiated to use this function"

        return self._model

    @property
    def model_has_been_trained(self) -> bool:
        if self._model is None:
            return False
        else:
            return True

    def _train_backwards(self) -> None:

        assert self._model is not None, "Model must be initialised to use this function"
        assert self._marginal_log_likelihood is not None, "Model must be initialised to use this function"

        assert self._model_optimiser.optimiser is not None, "Model optimiser must be initiated to use this function"

        loss_difference = torch.tensor(1e5)  # initial values
        loss = torch.tensor(1e20)  # TODO: Find a way to make sure this number is always big enough
        assert self.training_settings.loss_change_to_stop < loss_difference
        iteration = 1

        while bool(loss_difference > self.training_settings.loss_change_to_stop):

            self._model_optimiser.optimiser.zero_grad()

            output = self._model(*self._model.train_inputs)

            previous_loss = loss
            loss = -self._marginal_log_likelihood(  # type: ignore  # gpytorch seems to be missing type-hints
                output,
                self._model.train_targets
            )

            loss.backward()
            loss_difference = torch.abs(previous_loss - loss)

            self._model_optimiser.optimiser.step()

            if self.training_settings.verbose:
                print(
                    f"Training model... Iteration {iteration} (of a maximum {self.training_settings.max_iter})"
                    f" - MLL: {loss.item():.3f}",
                    end="\r"
                )

            iteration += 1
            if iteration > self.training_settings.max_iter:
                warnings.warn("Stopped training due to maximum iterations reached.")
                break

        if self.training_settings.verbose:
            print("\n")

    def _initiate_optimiser(self) -> None:

        parameters: list[dict[str, Iterator[torch.nn.Parameter]]] = []
        for model in self._model_list:
            parameters += model.trained_parameters

        self._model_optimiser.initiate_optimiser(
            parameters=parameters
        )

    def _set_mode_evaluate(self) -> None:

        assert self._model is not None, "Model must be initialised to set its mode."
        assert self._likelihood is not None, "Model must be initialised to set its mode."

        self._model.eval()
        self._likelihood.eval()

    def _set_mode_train(self) -> None:

        assert self._model is not None, "Model must be initialised to set its mode."
        assert self._likelihood is not None, "Model must be initialised to set its mode."

        self._model.train()
        self._likelihood.train()

    def _set_mode(
            self,
            model_mode: ModelMode
    ) -> None:

        if model_mode == ModelMode.evaluating:
            self._set_mode_evaluate()

        elif model_mode == ModelMode.training:
            self._set_mode_train()

    @property
    def _mode(self) -> ModelMode:

        assert self._model is not None, "Model must be initialised to get its mode."

        if self._model.training:
            return ModelMode.training

        else:
            return ModelMode.evaluating

    @property
    def multi_objective(self) -> bool:

        if self.n_objectives > 1:
            return True
        else:
            return False
