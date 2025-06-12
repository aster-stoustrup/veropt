import abc
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, Optional, TypedDict, Union, Unpack

import botorch
import gpytorch
import torch
from gpytorch.constraints import GreaterThan, Interval, LessThan
from gpytorch.distributions import MultivariateNormal

from veropt.optimiser.utility import check_variable_and_objective_shapes, check_variable_objective_values_matching, \
    unpack_variables_objectives_from_kwargs


# TODO: Consider deleting this abstraction. Does it have a function at this point?
class SurrogateModel:
    __metaclass__ = abc.ABCMeta

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
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:
        pass


class GPyTorchDataModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
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


class GPyTorchSingleModel:

    def __init__(
            self,
            likelihood: gpytorch.likelihoods.GaussianLikelihood,
            mean_module: gpytorch.means.Mean,
            kernel: gpytorch.kernels.Kernel,
    ) -> None:

        self.likelihood = likelihood
        self.mean_module = mean_module
        self.kernel = kernel

        self.model_with_data: GPyTorchDataModel | None = None

        self.trained_parameters: list[dict[str, Iterator[torch.nn.Parameter]]] = [{}]

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
class MaternParameters:
    lengthscale_lower_bound: float = 0.1
    lengthscale_upper_bound: float = 2.0
    noise: float = 1e-8
    noise_lower_bound: float = 1e-8
    train_noise: bool = False


class MaternSingleModel(GPyTorchSingleModel):
    def __init__(
            self,
            n_variables: int,
            **kwargs: Unpack[MaternParametersInputDict]
    ) -> None:

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        mean_module = gpytorch.means.ConstantMean()
        kernel = gpytorch.kernels.MaternKernel(
            ard_num_dims=n_variables,
            batch_shape=torch.Size([])
        )

        self.settings = MaternParameters(
            **kwargs
        )

        super().__init__(
            likelihood=likelihood,
            mean_module=mean_module,
            kernel=kernel
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

    def initialise_model_with_data(
            self,
            train_inputs: torch.Tensor,
            train_targets: torch.Tensor,
    ) -> None:

        super().initialise_model_with_data(
            train_inputs=train_inputs,
            train_targets=train_targets
        )

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

        self._set_up_trained_parameters()

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

    def get_lengthscale(self) -> float:

        assert self.model_with_data is not None, "Must have trained model before calling this"

        return float(self.model_with_data.covar_module.lengthscale)

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


class TorchModelOptimiser:
    def __init__(
            self,
            optimiser_class: type[torch.optim.Optimizer],
            optimiser_settings: Optional[dict] = None
    ) -> None:
        self.optimiser: Optional[torch.optim.Optimizer] = None
        self.optimiser_class = optimiser_class

        self.optimiser_settings = optimiser_settings or {}

    def initiate_optimiser(
            self,
            parameters: Iterator[torch.nn.Parameter] | list[dict[str, Iterator[torch.nn.Parameter]]]
    ) -> None:

        self.optimiser = self.optimiser_class(
            params=parameters,
            **self.optimiser_settings
        )


class AdamModelOptimiser(TorchModelOptimiser):
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
    init_max_iter: int


@dataclass
class GPyTorchTrainingParameters:
    learning_rate: float = 0.1
    loss_change_to_stop: float = 1e-6  # TODO: Find optimal value for this?
    max_iter: int = 1000
    init_max_iter: int = 10000


class GPyTorchFullModel(SurrogateModel):

    def __init__(
            self,
            n_variables: int,
            n_objectives: int,
            single_model_list: list[GPyTorchSingleModel],
            model_optimiser: TorchModelOptimiser,
            verbose: bool = True,
            **kwargs: Unpack[GPyTorchTrainingParametersInputDict]
    ) -> None:

        self.training_parameters = GPyTorchTrainingParameters(
            **kwargs
        )

        assert len(single_model_list) == n_objectives, "Number of objectives must match the length of the model list"

        self._model_list = single_model_list
        self._model: Optional[botorch.models.ModelListGP] = None
        self._likelihood: Optional[gpytorch.likelihoods.LikelihoodList] = None
        self._marginal_log_likelihood: Optional[gpytorch.mlls.SumMarginalLogLikelihood] = None

        self._model_optimiser = model_optimiser

        self.verbose = verbose

        super().__init__(
            n_variables=n_variables,
            n_objectives=n_objectives
        )

    @staticmethod
    def _check_input_dimensions[T, **P](
            function: Callable[P, T]
    ) -> Callable[P, T]:

        def check_dimensions(
                *args: P.args,
                **kwargs: P.kwargs,
        ) -> T:

            self = args[0]
            assert type(self) is GPyTorchFullModel

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

    @_check_input_dimensions
    def __call__(
            self,
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

    @check_variable_objective_values_matching
    @_check_input_dimensions
    def train_model(
            self,
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
            variable_values: torch.Tensor,
            objective_values: torch.Tensor
    ) -> None:

        for objective_number in range(self.n_objectives):

            self._model_list[objective_number].initialise_model_with_data(
                train_inputs=variable_values,
                train_targets=objective_values[:, objective_number]
            )

        for model in self._model_list:
            assert model.model_with_data is not None, "Model initialisation seems to have failed"

        # TODO: Might need to look into more options here
        #   - Currently seems to be assuming independent models. Maybe need to add an option for this?
        #       - Probably would need a different class (GPyTorchIndependentModels vs the opposite)
        #       - Botorch has some options for this
        self._model = botorch.models.ModelListGP(
            *[model.model_with_data for model in self._model_list]  # type: ignore
        )
        self._likelihood = gpytorch.likelihoods.LikelihoodList(
            *[model.model_with_data.likelihood for model in self._model_list]  # type: ignore[union-attr]
        )

    def get_gpytorch_model(self) -> botorch.models.ModelListGP:

        assert self._model is not None, "Model must be initiated to use this function"

        return self._model

    def _train_backwards(self) -> None:

        assert self._model is not None, "Model must be initialised to use this function"
        assert self._marginal_log_likelihood is not None, "Model must be initialised to use this function"

        assert self._model_optimiser.optimiser is not None, "Model optimiser must be initiated to use this function"

        loss_difference = torch.tensor(1e5)  # initial values
        loss = torch.tensor(1e20)  # TODO: Find a way to make sure this number is always big enough
        assert self.training_parameters.loss_change_to_stop < loss_difference
        iteration = 1

        while bool(loss_difference > self.training_parameters.loss_change_to_stop):

            self._model_optimiser.optimiser.zero_grad()

            output = self._model(*self._model.train_inputs)

            previous_loss = loss
            # Ignoring mypy here because gpytorch seems to be missing type-hints
            loss = -self._marginal_log_likelihood(  # type: ignore
                output,
                self._model.train_targets
            )

            loss.backward()
            loss_difference = torch.abs(previous_loss - loss)

            self._model_optimiser.optimiser.step()

            if self.verbose:
                print(
                    f"Training model... Iteration {iteration} (of a maximum {self.training_parameters.max_iter})"
                    f" - Loss: {loss.item():.3f}",
                    end="\r"
                )

            iteration += 1
            if iteration > self.training_parameters.max_iter:
                warnings.warn("Stopped training due to maximum iterations reached.")
                break

        if self.verbose:
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


# TODO: Implement default model (pre-sets)
