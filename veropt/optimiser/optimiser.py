from copy import deepcopy
from functools import cached_property
from inspect import get_annotations
from typing import Optional, Union, Unpack

import gpytorch.settings
import torch

from veropt.optimiser.acquisition import AcquisitionFunction
from veropt.optimiser.model import SurrogateModel
from veropt.optimiser.normaliser import Normaliser
from veropt.optimiser.objective import IntegratedObjective, InterfaceObjective, ObjectiveKind, determine_objective_type
from veropt.optimiser.optimiser_utility import (
    DataShape, InitialPointsGenerationMode, OptimisationMode,
    OptimiserSettings, OptimiserSettingsInputDict, SuggestedPoints, TensorWithNormalisationFlag, format_list,
    get_best_points, get_pareto_optimal_points
)


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
            acquisition_function: AcquisitionFunction,
            model: SurrogateModel,
            normaliser_class: type[Normaliser],
            **kwargs: Unpack[OptimiserSettingsInputDict]
    ):

        self.objective = objective
        self.n_objectives = objective.n_objectives
        self.bounds_real_units = objective.bounds

        self.acquisition_function = acquisition_function
        self.model = model
        self.normaliser_class = normaliser_class

        self._normaliser_variables: Optional[Normaliser] = None
        self._normaliser_values: Optional[Normaliser] = None

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
        self.evaluated_variables_real_units: torch.Tensor = torch.Tensor()
        self.evaluated_objective_real_units: torch.Tensor = torch.Tensor()

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

            suggested_variables = self.initial_points[
                    self.n_points_evaluated: self.n_points_evaluated + self.n_evaluations_per_step
                ]

        elif self.optimisation_mode == OptimisationMode.bayesian:

            suggested_variables = self._find_candidates_with_model()

        else:
            raise RuntimeError

        if self.model_has_been_trained:

            predicted_values_tensor = self.model(suggested_variables.tensor)
            predicted_values = TensorWithNormalisationFlag(
                tensor=predicted_values_tensor,
                normalised=self.return_normalised_data
            )

        else:

            predicted_values = None

        self.suggested_points = SuggestedPoints(
            variable_values=suggested_variables,
            predicted_objective_values=predicted_values,
            generated_at_step=self.current_step
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

        new_variables_real_units = self._unnormalise_variables(self.suggested_points.variable_values)

        objective_function_values = self.objective.run(new_variables_real_units.tensor) # type: ignore[union-attr]

        self._reset_suggested_points()

        return (
            new_variables_real_units,
            TensorWithNormalisationFlag(
                tensor=objective_function_values,
                normalised=False
            )
        )

    def _add_new_points(
            self,
            new_variables: TensorWithNormalisationFlag,
            new_values: TensorWithNormalisationFlag
    ) -> None:

        assert new_variables.normalised is False
        assert new_values.normalised is False

        assert new_variables.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step
        assert new_values.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step

        if self.n_points_evaluated == 0:

            self.evaluated_variables_real_units = new_variables.tensor.detach()
            self.evaluated_objective_real_units = new_values.tensor.detach()

        else:

            self.evaluated_variables_real_units = torch.cat(
                tensors=[self.evaluated_variables_real_units, new_variables.tensor.detach()],
                dim=DataShape.index_points
            )
            self.evaluated_objective_real_units = torch.cat(
                tensors=[self.evaluated_objective_real_units, new_values.tensor.detach()],
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

    def _load_latest_points(self):

        assert self._objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to load points."
        )

        # TODO: Implement
        #   - Add type hints
        # self.objective.load_evaluated_points()

        raise NotImplementedError

    def _save_candidates(self):

        assert self._objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to save candidates."
        )

        # TODO: Implement
        #   - Add type hints
        # self.objective.save_candidates()

        raise NotImplementedError

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

    def _find_candidates_with_model(self) -> TensorWithNormalisationFlag:

        self._refresh_acquisition_function()

        if self.settings.verbose:
            print("Finding candidates for the next points to evaluate...")

        suggested_variables = TensorWithNormalisationFlag(
            tensor=self.acquisition_function.suggest_points(),
            normalised=self.return_normalised_data
        )

        if self.settings.verbose:
            print(f"Found all {self.n_evaluations_per_step} candidates.")

        return suggested_variables

    def _refresh_acquisition_function(self) -> None:
        raise NotImplementedError

    def _reset_suggested_points(self) -> None:
        self.suggested_points_history[self.current_step - 1] = deepcopy(self.suggested_points)
        self.suggested_points = None

    def _train_model(self) -> None:

        if self.settings.normalise:
            assert self.data_has_been_normalised

        self.model.train_model(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor
        )
        self._refresh_acquisition_function()

        self.model_has_been_trained = True

        self.suggested_points = None

    def _fit_normaliser(self) -> None:

        self._normaliser_variables = self.normaliser_class(
            tensor=self.evaluated_variables_real_units
        )
        self._normaliser_values = self.normaliser_class(
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

        self.acquisition_function.set_bounds(self.bounds.tensor)

        if self.settings.verbose:
            # TODO: Find a way to do this. Don't want it to print this every time.
            raise NotImplementedError
            # print("Normalisation has been completed.")

    def _unnormalise_variables(
            self,
            variable_values: TensorWithNormalisationFlag
    ) -> TensorWithNormalisationFlag:

        assert self._normaliser_variables is not None, "Normaliser must be initialised to do this action"

        if variable_values.normalised:
            variables_real_units = self._normaliser_variables.inverse_transform(variable_values.tensor)
        else:
            variables_real_units = variable_values.tensor

        return TensorWithNormalisationFlag(
            tensor=variables_real_units,
            normalised=False
        )

    def _print_status(self) -> None:

        best_variables, best_values, max_index = self.get_best_points()

        assert best_variables is not None, "Failed to get best points"
        assert best_values is not None, "Failed to get best points"
        assert max_index is not None, "Failed to get best points"

        best_values_string = format_list(best_values.tolist())

        best_values_variables_string = format_list(best_variables.tolist())

        newest_value_string = format_list(self.evaluated_objective_values[:, -1].tensor.tolist())

        newest_variables_string = format_list(self.evaluated_variable_values[:, -1].tensor.tolist())

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

        assert self._normaliser_values is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_values.transform(self.evaluated_objective_real_units)

    @cached_property
    def bounds_normalised(self) -> torch.Tensor:

        assert self._normaliser_variables is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_variables.transform(self.bounds_real_units)

    @cached_property
    def initial_points_normalised(self) -> torch.Tensor:

        assert self._normaliser_variables is not None, "Normaliser must be initiated to get these values"

        return self._normaliser_variables.transform(self.initial_points_real_units)
