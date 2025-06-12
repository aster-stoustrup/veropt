from copy import deepcopy
from functools import cached_property
from inspect import get_annotations
from typing import Callable, Optional, Union, Unpack

import gpytorch.settings
import torch

from veropt.optimiser.normaliser import Normaliser
from veropt.optimiser.objective import IntegratedObjective, InterfaceObjective, ObjectiveKind, determine_objective_type
from veropt.optimiser.optimiser_utility import (
    DataShape, InitialPointsGenerationMode, OptimisationMode,
    OptimiserSettings, OptimiserSettingsInputDict, SuggestedPoints,
    TensorWithNormalisationFlag, format_input_from_objective,
    format_output_for_objective, get_best_points, get_pareto_optimal_points,
    list_with_floats_to_string
)
from veropt.optimiser.prediction import Predictor
from veropt.optimiser.utility import check_variable_and_objective_shapes, \
    unpack_flagged_variables_objectives_from_kwargs


def generate_initial_points_random(
        bounds: torch.Tensor,
        n_initial_points: int,
        n_variables: int
) -> torch.Tensor:

    return (bounds[1] - bounds[0]) * torch.rand(n_initial_points, n_variables) + bounds[0]


class BayesianOptimiser:
    def __init__(
            self,
            n_initial_points: int,
            n_bayesian_points: int,
            objective: Union[IntegratedObjective, InterfaceObjective],
            predictor: Predictor,
            normaliser_class: type[Normaliser],
            **kwargs: Unpack[OptimiserSettingsInputDict]
    ):

        self.objective = objective
        self.n_objectives = objective.n_objectives
        self.bounds_real_units = objective.bounds

        self.predictor = predictor
        self.normaliser_class = normaliser_class

        self._normaliser_variables: Optional[Normaliser] = None
        self._normaliser_objective_values: Optional[Normaliser] = None

        # TODO: Move this assert somewhere else?
        # TODO: Write error message for this assert
        for key in kwargs.keys():
            assert key in get_annotations(OptimiserSettingsInputDict).keys()

        self.settings = OptimiserSettings(
            n_initial_points=n_initial_points,
            n_bayesian_points=n_bayesian_points,
            n_objectives=self.n_objectives,
            **kwargs
        )

        self._set_up_settings()

        self.initial_points_real_units = self._generate_initial_points()
        self.evaluated_variables_real_units: torch.Tensor = torch.tensor([])
        self.evaluated_objective_real_units: torch.Tensor = torch.tensor([])

        self.data_has_been_normalised = False
        self.model_has_been_trained = False

        self.suggested_points: Optional[SuggestedPoints] = None
        self.suggested_points_history: list[Optional[SuggestedPoints]] = (
                [None] * (self.n_initial_points + self.n_bayesian_points)
        )

        self._objective_type = determine_objective_type(
            objective=objective
        )

        self._verify_set_up()

    @staticmethod
    def _check_input_dimensions[T, **P](
            function: Callable[P, T]
    ) -> Callable[P, T]:

        def check_dimensions(
                *args: P.args,
                **kwargs: P.kwargs,
        ) -> T:

            self = args[0]
            assert type(self) is BayesianOptimiser

            variable_values, objective_values = unpack_flagged_variables_objectives_from_kwargs(kwargs)

            if variable_values is None and objective_values is None:
                raise RuntimeError("This decorator was called to check input shapes but found no valid inputs.")

            check_variable_and_objective_shapes(
                n_variables=self.objective.n_variables,
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

    def run_optimisation_step(self) -> None:

        if self._objective_type == ObjectiveKind.integrated:

            self.suggest_candidates()

            new_variables, new_values = self._evaluate_points()

            self._add_new_points(new_variables, new_values)

        elif self._objective_type == ObjectiveKind.interface:

            self._load_latest_points()

            self.suggest_candidates()

            self._save_candidates()

    def suggest_candidates(self) -> None:

        if self.optimisation_mode == OptimisationMode.initial:

            suggested_variables_tensor = self.initial_points[
                    self.n_points_evaluated: self.n_points_evaluated + self.n_evaluations_per_step
                ].tensor

            prediction = None

        elif self.optimisation_mode == OptimisationMode.bayesian:

            suggested_variables_tensor = self.predictor.suggest_points(
                verbose=self.settings.verbose
            )

            if self.model_has_been_trained:

                prediction = self.predictor.predict_values(
                    variable_values=suggested_variables_tensor
                )

            else:

                prediction = None

        else:
            raise RuntimeError


        self.suggested_points = SuggestedPoints(
            variable_values=suggested_variables_tensor,
            predicted_objective_values=prediction,
            generated_at_step=deepcopy(self.current_step),
            normalised= deepcopy(self.return_normalised_data)
        )

    def get_best_points(self) -> tuple[torch.Tensor, torch.Tensor, int] | tuple[None, None, None]:

        best_variables, best_values, max_index = get_best_points(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor,
            weights=self.settings.objective_weights
        )

        return (
            best_variables,
            best_values,
            max_index
        )

    def get_pareto_optimal_points(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:

        pareto_variables, pareto_values, pareto_indices = get_pareto_optimal_points(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor,
            weights=self.settings.objective_weights
        )

        return (
            pareto_variables,
            pareto_values,
            pareto_indices
        )

    def _evaluate_points(self) -> tuple[TensorWithNormalisationFlag, TensorWithNormalisationFlag]:

        assert self._objective_type == ObjectiveKind.integrated, (
            "The objective must be an 'IntegratedObjective' to be evaluated during optimisation."
        )

        assert self.suggested_points is not None, "Suggested points must be created before using this function"

        new_variables_real_units = self._unnormalise_variables(self.suggested_points.variable_values_flagged)

        objective_function_values = self.objective.run(new_variables_real_units.tensor)  # type: ignore[union-attr]

        self._reset_suggested_points()

        return (
            new_variables_real_units,
            TensorWithNormalisationFlag(
                tensor=objective_function_values,
                normalised=False
            )
        )

    @_check_input_dimensions
    def _add_new_points(
            self,
            variable_values_flagged: TensorWithNormalisationFlag,
            objective_values_flagged: TensorWithNormalisationFlag
    ) -> None:

        assert variable_values_flagged.normalised is False
        assert objective_values_flagged.normalised is False

        # TODO: Write good error message
        #   - Could also move this check somewhere...?
        #   - Then again, maybe we want a more flexible way to handle this in the future...?
        assert variable_values_flagged.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step
        assert objective_values_flagged.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step

        if self.n_points_evaluated == 0:

            self.evaluated_variables_real_units = variable_values_flagged.tensor.detach()
            self.evaluated_objective_real_units = objective_values_flagged.tensor.detach()

        else:

            self.evaluated_variables_real_units = torch.cat(
                tensors=[self.evaluated_variables_real_units, variable_values_flagged.tensor.detach()],
                dim=DataShape.index_points
            )
            self.evaluated_objective_real_units = torch.cat(
                tensors=[self.evaluated_objective_real_units, objective_values_flagged.tensor.detach()],
                dim=DataShape.index_points
            )

        if self.model_has_been_trained:

            if self.settings.normalise:

                if self.settings.renormalise_each_step:

                    self._fit_normaliser()
                    self._update_normalised_values()

                else:

                    self._update_normalised_values()

            self._train_model()

        elif self.n_points_evaluated >= self.settings.n_points_before_fitting:

            if self.settings.normalise:

                self._fit_normaliser()
                self._update_normalised_values()

            self._train_model()

        if self.settings.verbose:
            self._print_status()

    def _load_latest_points(self) -> None:

        assert self._objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to load points."
        )

        (new_variable_values, new_objective_values) = self.objective.load_evaluated_points()  # type: ignore[union-attr]

        new_variable_values_tensor, new_objective_values_tensor = format_input_from_objective(
            new_variable_values=new_variable_values,
            new_objective_values=new_objective_values,
            variable_names=self.objective.variable_names,
            objective_names=self.objective.objective_names,
            expected_amount_points=self.n_evaluations_per_step
        )

        self._add_new_points(
            variable_values_flagged=TensorWithNormalisationFlag(
                tensor=new_variable_values_tensor,
                normalised=False
            ),
            objective_values_flagged=TensorWithNormalisationFlag(
                tensor=new_objective_values_tensor,
                normalised=False
            )
        )

    def _save_candidates(self) -> None:

        assert self._objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to save candidates."
        )

        assert self.suggested_points is not None, "Must have made suggestions before saving them."

        suggested_variables_real_units = self._unnormalise_variables(self.suggested_points.variable_values_flagged)

        suggested_variables_dict = format_output_for_objective(
            suggested_variables=suggested_variables_real_units.tensor,
            variable_names=self.objective.variable_names
        )

        self.objective.save_candidates(  # type: ignore[union-attr]
            suggested_variables=suggested_variables_dict
        )

    def _verify_set_up(self) -> None:

        assert self.n_initial_points % self.n_evaluations_per_step == 0, (
            "The amount of initial points is not divisable by the amount of points evaluated each step."
        )

        assert self.n_bayesian_points % self.n_evaluations_per_step == 0, (
            "The amount of bayesian points is not divisable by the amount of points evaluated each step."
        )

    def _set_up_settings(self) -> None:

        if self.settings.mask_nans:
            raise NotImplementedError(
                "Make a test to see if this works. Otherwise, might need to use as a context manager during training?"
            )
            gpytorch.settings.observation_nan_policy('mask')

    def _generate_initial_points(self) -> torch.Tensor:

        if self.settings.initial_points_generator == InitialPointsGenerationMode.random:
            return generate_initial_points_random(
                bounds=self.objective.bounds,
                n_initial_points=self.n_initial_points,
                n_variables=self.objective.n_variables
            )

        else:
            raise ValueError(
                f"Initial point mode {self.settings.initial_points_generator} is not understood or not implemented."
            )

    def _reset_suggested_points(self) -> None:
        self.suggested_points_history[self.current_step - 1] = deepcopy(self.suggested_points)
        self.suggested_points = None

    def _train_model(self) -> None:

        if self.settings.normalise:
            assert self.data_has_been_normalised

        self.predictor.update_with_new_data(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor
        )

        self.model_has_been_trained = True

        self.suggested_points = None

    def _fit_normaliser(self) -> None:

        self._normaliser_variables = self.normaliser_class(
            tensor=self.evaluated_variables_real_units
        )
        self._normaliser_objective_values = self.normaliser_class(
            tensor=self.evaluated_objective_real_units
        )

        self.data_has_been_normalised = True

    def _update_normalised_values(self) -> None:

        cached_normalised_values = [
            'evaluated_variables_normalised',
            'evaluated_variables_values_normalised',
            'bounds_normalised',
            'initial_points_normalised'
        ]

        # TODO: Write a test for this
        for normalised_value in cached_normalised_values:
            del self.__dict__[normalised_value]

        self.predictor.update_bounds(self.bounds.tensor)

        if self.settings.verbose:
            # TODO: Find a way to do this. Don't want it to print this every time.
            raise NotImplementedError
            # print("Normalisation has been completed.")

    @_check_input_dimensions
    def _unnormalise_variables(
            self,
            variable_values_flagged: TensorWithNormalisationFlag
    ) -> TensorWithNormalisationFlag:

        assert self._normaliser_variables is not None, "Normaliser must be initialised to do this action"

        if variable_values_flagged.normalised:
            variables_real_units = self._normaliser_variables.inverse_transform(variable_values_flagged.tensor)
        else:
            variables_real_units = variable_values_flagged.tensor

        return TensorWithNormalisationFlag(
            tensor=variables_real_units,
            normalised=False
        )

    def _print_status(self) -> None:

        best_variables, best_values, max_index = self.get_best_points()

        assert best_variables is not None, "Failed to get best points"
        assert best_values is not None, "Failed to get best points"
        assert max_index is not None, "Failed to get best points"

        best_values_string = list_with_floats_to_string(best_values.tolist())

        best_values_variables_string = list_with_floats_to_string(best_variables.tolist())

        newest_value_string = list_with_floats_to_string(self.evaluated_objective_values[:, -1].tensor.tolist())

        newest_variables_string = list_with_floats_to_string(self.evaluated_variable_values[:, -1].tensor.tolist())

        status_string = (
            f"Optimisation running in {self.optimisation_mode.name} mode "
            f"at step {self.current_step} out of {self.n_points_evaluated} \n"
            f"Best objective value(s): {best_values_string} at variable values {best_values_variables_string} \n"
            f"Newest objective value(s): {newest_value_string} at variable values {newest_variables_string} \n"
        )

        print(status_string)

    @property
    def n_initial_points(self) -> int:
        return self.settings.n_initial_points

    @property
    def n_bayesian_points(self) -> int:
        return self.settings.n_bayesian_points

    @property
    def n_evaluations_per_step(self) -> int:
        return self.settings.n_evaluations_per_step

    @property
    def current_step(self) -> int:
        assert self.n_points_evaluated % self.n_evaluations_per_step == 0, (
            "Amount of points evaluated does not match step size."
        )
        return self.n_points_evaluated // self.n_evaluations_per_step + 1

    @property
    def n_points_evaluated(self) -> int:
        return self.evaluated_variable_values.tensor.shape[DataShape.index_points]

    @property
    def optimisation_mode(self) -> OptimisationMode:
        if self.n_points_evaluated < self.n_initial_points:
            return OptimisationMode.initial
        else:
            return OptimisationMode.bayesian

    @property
    def return_normalised_data(self) -> bool:
        if self.settings.normalise and self.data_has_been_normalised:
            return True
        else:
            return False

    @property
    def evaluated_variable_values(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            variable_values = self.evaluated_variables_normalised
        else:
            variable_values = self.evaluated_variables_real_units

        return TensorWithNormalisationFlag(
            tensor=variable_values,
            normalised=self.return_normalised_data
        )

    @property
    def evaluated_objective_values(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            values = self.evaluated_objective_normalised
        else:
            values = self.evaluated_objective_real_units

        return TensorWithNormalisationFlag(
            tensor=values,
            normalised=self.return_normalised_data
        )

    @property
    def bounds(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            bounds = self.bounds_normalised
        else:
            bounds = self.objective.bounds

        return TensorWithNormalisationFlag(
            tensor=bounds,
            normalised=self.return_normalised_data
        )

    @property
    def initial_points(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            initial_points = self.initial_points_normalised
        else:
            initial_points = self.initial_points_real_units

        return TensorWithNormalisationFlag(
            tensor=initial_points,
            normalised=self.return_normalised_data
        )

    @cached_property
    def evaluated_variables_normalised(self) -> torch.Tensor:

        assert self._normaliser_variables is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_variables.transform(self.evaluated_variables_real_units)

    @cached_property
    def evaluated_objective_normalised(self) -> torch.Tensor:

        assert self._normaliser_objective_values is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_objective_values.transform(self.evaluated_objective_real_units)

    @cached_property
    def bounds_normalised(self) -> torch.Tensor:

        assert self._normaliser_variables is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_variables.transform(self.bounds_real_units)

    @cached_property
    def initial_points_normalised(self) -> torch.Tensor:

        assert self._normaliser_variables is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_variables.transform(self.initial_points_real_units)
