from copy import deepcopy
from functools import cached_property
from inspect import get_annotations
from typing import Optional, Union, Unpack

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
            acquisition_function: AcquisitionFunction = None,
            model: SurrogateModel = None,
            normaliser_class: type[Normaliser] = None,
            **kwargs: Unpack[OptimiserSettingsInputDict]
    ):

        self.objective = objective
        self.bounds_real_units = objective.bounds

        # TODO: Consider making methods for these to keep init as clean as possible?
        if acquisition_function is None:
            raise NotImplementedError
        else:
            self.acquisition_function = acquisition_function

        if model is None:
            raise NotImplementedError
        else:
            self.model = model

        if normaliser_class is None:
            raise NotImplementedError
        else:
            self.normaliser_class = normaliser_class

        self.normaliser_variables = None
        self.normaliser_values = None

        # TODO: Move this assert somewhere else?
        # TODO: Write error message for this assert
        for key in kwargs.keys():
            assert key in get_annotations(OptimiserSettingsInputDict).keys()

        self.settings = OptimiserSettings(
            n_initial_points=n_initial_points,
            n_bayesian_points=n_bayesian_points,
            **kwargs
        )

        self.initial_points_real_units = self._generate_initial_points()
        self.evaluated_variables_real_units = None
        self.evaluated_values_real_units = None

        self.data_has_been_normalised = False
        self.model_has_been_trained = False

        self.suggested_points: Optional[SuggestedPoints] = None
        self.suggested_points_history: list[Optional[SuggestedPoints]] = (
                [None] * (self.n_initial_points + self.n_bayesian_points)
        )

        self.objective_type = determine_objective_type(
            objective=objective
        )

        self._verify_set_up()

    def run_optimisation_step(self):

        if self.objective_type == ObjectiveKind.integrated:

            self.suggest_candidates()

            new_variables, new_values = self._evaluate_points()

            self._add_new_points(new_variables, new_values)

        elif self.objective_type == ObjectiveKind.interface:

            self._load_latest_points()

            self.suggest_candidates()

            self._save_candidates()

    def suggest_candidates(self):

        if self.optimisation_mode == OptimisationMode.initial:

            suggested_variables = self.initial_points[
                    self.n_points_evaluated: self.n_points_evaluated + self.n_evaluations_per_step
                ]

        elif self.optimisation_mode == OptimisationMode.bayesian:

            suggested_variables = self._find_candidates_with_model()

        else:
            raise RuntimeError

        if self.model_has_been_trained:
            predicted_values = self.model(suggested_variables.tensor)
            predicted_values = TensorWithNormalisationFlag(
                tensor=predicted_values,
                normalised=self.return_normalised_data
            )
        else:
            predicted_values = None

        self.suggested_points = SuggestedPoints(
            variables=suggested_variables,
            predicted_values=predicted_values,
            generated_at_step=self.current_step
        )

    def get_best_points(self) -> (torch.Tensor, torch.Tensor, int):

        best_variables, best_values, max_index = get_best_points(
            variables=self.evaluated_variables.tensor,
            values=self.evaluated_values.tensor,
            weights=self.settings.objective_weights
        )

        return (
            best_variables,
            best_values,
            max_index
        )

    def get_pareto_optimal_points(self) -> (torch.Tensor, torch.Tensor, list[bool]):

        pareto_variables, pareto_values, pareto_indices = get_pareto_optimal_points(
            variables=self.evaluated_variables.tensor,
            values=self.evaluated_values.tensor,
            weights=self.settings.objective_weights
        )

        return (
            pareto_variables,
            pareto_values,
            pareto_indices
        )

    def _evaluate_points(self) -> (TensorWithNormalisationFlag, TensorWithNormalisationFlag):

        assert self.objective_type == ObjectiveKind.integrated, (
            "The objective must be an 'IntegratedObjective' to be evaluated during optimisation."
        )

        new_variables_real_units = self._unnormalise_variables(self.suggested_points.variables)

        objective_function_values = self.objective.run(new_variables_real_units.tensor)

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
    ):

        assert new_variables.normalised is False
        assert new_values.normalised is False

        assert new_variables.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step
        assert new_values.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step

        if self.n_points_evaluated == 0:

            self.evaluated_variables_real_units = new_variables.tensor.detach()
            self.evaluated_values_real_units = new_values.tensor.detach()

        else:

            self.evaluated_variables_real_units = torch.cat(
                tensors=[self.evaluated_variables_real_units, new_variables.tensor.detach()],
                dim=DataShape.index_points
            )
            self.evaluated_values_real_units = torch.cat(
                tensors=[self.evaluated_values_real_units, new_values.tensor.detach()],
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

        assert self.objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to load points."
        )

        # TODO: Implement
        # self.objective.load_evaluated_points()

        raise NotImplementedError

    def _save_candidates(self):

        assert self.objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to save candidates."
        )

        # TODO: Implement
        # self.objective.save_candidates()

        raise NotImplementedError

    def _verify_set_up(self):

        assert self.n_initial_points % self.n_evaluations_per_step == 0, (
            "The amount of initial points is not divisable by the amount of points evaluated each step."
        )

        assert self.n_bayesian_points % self.n_evaluations_per_step == 0, (
            "The amount of bayesian points is not divisable by the amount of points evaluated each step."
        )

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

    def _refresh_acquisition_function(self):
        raise NotImplementedError

    def _reset_suggested_points(self):
        self.suggested_points_history[self.current_step - 1] = deepcopy(self.suggested_points)
        self.suggested_points = None

    def _train_model(self):

        if self.settings.normalise:
            assert self.data_has_been_normalised

        self.model.train_model(
            variables=self.evaluated_variables.tensor,
            values=self.evaluated_values.tensor
        )
        self._refresh_acquisition_function()

        self.model_has_been_trained = True

        self.suggested_points = None

    def _fit_normaliser(self):

        self.normaliser_variables = self.normaliser_class(
            tensor=self.evaluated_variables_real_units
        )
        self.normaliser_values = self.normaliser_class(
            tensor=self.evaluated_values_real_units
        )

        self.data_has_been_normalised = True

    def _update_normalised_values(self):

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
            variables: TensorWithNormalisationFlag
    ) -> TensorWithNormalisationFlag:

        if variables.normalised:
            variables_real_units = self.normaliser_variables.inverse_transform(variables.tensor)
        else:
            variables_real_units = variables.tensor

        return TensorWithNormalisationFlag(
            tensor=variables_real_units,
            normalised=False
        )

    def _print_status(self):

        best_variables, best_values, max_index = self.get_best_points()

        best_values_string = format_list(best_values.tolist())

        best_values_variables_string = format_list(best_variables.tolist())

        newest_value_string = format_list(self.evaluated_values[:, -1].tensor.tolist())

        newest_variables_string = format_list(self.evaluated_variables[:, -1].tensor.tolist())

        status_string = (
            f"Optimisation running in {self.optimisation_mode.name} mode "
            f"at step {self.current_step} out of {self.n_points_evaluated} \n"
            f"Best value(s): {best_values_string} at variables {best_values_variables_string} \n"
            f"Newest value(s): {newest_value_string} at variables {newest_variables_string} \n"
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
    def current_step(self):
        assert self.n_points_evaluated % self.n_evaluations_per_step == 0, (
            "Amount of points evaluated does not match step size."
        )
        return self.n_points_evaluated // self.n_evaluations_per_step + 1

    @property
    def n_points_evaluated(self) -> int:
        return self.evaluated_variables.shape[DataShape.index_points]

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
    def evaluated_variables(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            variables = self.evaluated_variables_normalised
        else:
            variables = self.evaluated_variables_real_units

        return TensorWithNormalisationFlag(
            tensor=variables,
            normalised=self.return_normalised_data
        )

    @property
    def evaluated_values(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            values = self.evaluated_values_normalised
        else:
            values = self.evaluated_values_real_units

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
    def evaluated_variables_normalised(self) -> TensorWithNormalisationFlag:
        return self.normaliser_variables.transform(self.evaluated_variables_real_units)

    @cached_property
    def evaluated_values_normalised(self) -> TensorWithNormalisationFlag:
        return self.normaliser_values.transform(self.evaluated_values_real_units)

    @cached_property
    def bounds_normalised(self):
        return self.normaliser_variables.transform(self.bounds_real_units)

    @cached_property
    def initial_points_normalised(self):
        return self.normaliser_variables.transform(self.initial_points_real_units)
