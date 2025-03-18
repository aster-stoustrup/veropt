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
    InitialPointsGenerationMode, OptimisationMode,
    OptimiserSettings, OptimiserSettingsInputDict, SuggestedPoints, TensorWithNormalisationFlag
)


def generate_initial_points_random(
        bounds: torch.Tensor,
        n_initial_points: int,
        n_parameters: int
) -> torch.Tensor:

    return (bounds[1] - bounds[0]) * torch.rand(n_initial_points, n_parameters) + bounds[0]


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

        self.normaliser_coordinates = None
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

        self.initial_points_real_units = self.generate_initial_points()
        self.evaluated_coordinates_real_units = None
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

        self.verify_set_up()

    def verify_set_up(self):

        assert self.n_initial_points % self.n_evaluations_per_step == 0, (
            "The amount of initial points is not divisable by the amount of points evaluated each step."
        )

        assert self.n_bayesian_points % self.n_evaluations_per_step == 0, (
            "The amount of bayesian points is not divisable by the amount of points evaluated each step."
        )

    def generate_initial_points(self) -> torch.Tensor:

        if self.settings.initial_points_generator == InitialPointsGenerationMode.random:
            return generate_initial_points_random(
                bounds=self.objective.bounds,
                n_initial_points=self.n_initial_points,
                n_parameters=self.objective.n_parameters
            )

        else:
            raise ValueError(
                f"Initial point mode {self.settings.initial_points_generator} is not understood or not implemented."
            )

    def run_optimisation_step(self):

        if self.objective_type == ObjectiveKind.integrated:

            self.suggest_candidates()

            new_coordinates, new_values = self.evaluate_points()

            self.add_new_points(new_coordinates, new_values)

        elif self.objective_type == ObjectiveKind.interface:

            self.load_latest_points()

            self.suggest_candidates()

            self.save_candidates()

    def evaluate_points(self) -> (TensorWithNormalisationFlag, TensorWithNormalisationFlag):

        assert self.objective_type == ObjectiveKind.integrated, (
            "The objective must be an 'IntegratedObjective' to be evaluated during optimisation."
        )

        new_coordinates_real_units = self.unnormalise_coordinates(self.suggested_points.coordinates)

        objective_function_values = self.objective.run(new_coordinates_real_units.tensor)

        self.reset_suggested_points()

        return (
            new_coordinates_real_units,
            TensorWithNormalisationFlag(
                tensor=objective_function_values,
                normalised=False
            )
        )

    def suggest_candidates(self):

        if self.optimisation_mode == OptimisationMode.initial:

            suggested_coordinates = self.initial_points[
                    self.n_points_evaluated: self.n_points_evaluated + self.n_evaluations_per_step
                ]

        elif self.optimisation_mode == OptimisationMode.bayesian:

            suggested_coordinates = self.find_candidates_with_model()

        else:
            raise RuntimeError

        if self.model_has_been_trained:
            predicted_values = self.model(suggested_coordinates.tensor)
            predicted_values = TensorWithNormalisationFlag(
                tensor=predicted_values,
                normalised=self.return_normalised_data
            )
        else:
            predicted_values = None

        self.suggested_points = SuggestedPoints(
            coordinates=suggested_coordinates,
            predicted_values=predicted_values,
            generated_at_step=self.current_step
        )

    def find_candidates_with_model(self) -> TensorWithNormalisationFlag:

        self.refresh_acquisition_function()

        if self.settings.verbose:
            print("Finding candidates for the next points to evaluate...")

        suggested_coordinates = TensorWithNormalisationFlag(
            tensor=self.acquisition_function.suggest_points(),
            normalised=self.return_normalised_data
        )

        if self.settings.verbose:
            print(f"Found all {self.n_evaluations_per_step} candidates.")

        return suggested_coordinates

    def refresh_acquisition_function(self):
        raise NotImplementedError

    def add_new_points(
            self,
            new_coordinates: TensorWithNormalisationFlag,
            new_values: TensorWithNormalisationFlag
    ):

        assert new_coordinates.normalised is False
        assert new_values.normalised is False

        assert new_coordinates.tensor.shape[1] == self.n_evaluations_per_step
        assert new_values.tensor.shape[1] == self.n_evaluations_per_step

        if self.n_points_evaluated == 0:

            self.evaluated_coordinates_real_units = new_coordinates.tensor
            self.evaluated_values_real_units = new_values.tensor

        else:

            self.evaluated_coordinates_real_units = torch.cat(
                tensors=[self.evaluated_coordinates_real_units, new_coordinates.tensor],
                dim=1
            )
            self.evaluated_values_real_units = torch.cat(
                tensors=[self.evaluated_values_real_units, new_values.tensor],
                dim=1
            )

        if self.model_has_been_trained:

            if self.settings.normalise:

                if self.settings.renormalise_each_step:

                    self.fit_normaliser()
                    self.update_normalised_values()

                else:

                    self.update_normalised_values()

            self.train_model()

        elif self.n_points_evaluated >= self.settings.n_points_before_fitting:

            if self.settings.normalise:

                self.fit_normaliser()
                self.update_normalised_values()

            self.train_model()

        if self.settings.verbose:
            self.print_status()

    def reset_suggested_points(self):
        self.suggested_points_history[self.current_step - 1] = deepcopy(self.suggested_points)
        self.suggested_points = None

    def train_model(self):

        if self.settings.normalise:
            assert self.data_has_been_normalised

        self.model.train_model(
            coordinates=self.evaluated_coordinates.tensor,
            values=self.evaluated_values.tensor
        )
        self.refresh_acquisition_function()

        self.model_has_been_trained = True

        self.suggested_points = None

    def fit_normaliser(self):

        self.normaliser_coordinates = self.normaliser_class(
            tensor=self.evaluated_coordinates_real_units
        )
        self.normaliser_values = self.normaliser_class(
            tensor=self.evaluated_values_real_units
        )

        self.data_has_been_normalised = True

    def update_normalised_values(self):

        cached_normalised_values = [
            'evaluated_coordinates_normalised',
            'evaluated_coordinates_values_normalised',
            'bounds_normalised',
            'initial_points_normalised'
        ]

        # TODO: Write a test for this
        for normalised_value in cached_normalised_values:
            del self.__dict__[normalised_value]

        self.acquisition_function.set_bounds(self.bounds)

        if self.settings.verbose:
            # TODO: Find a way to do this. Don't want it to print this every time.
            raise NotImplementedError
            # print("Normalisation has been completed.")

    def unnormalise_coordinates(
            self,
            coordinates: TensorWithNormalisationFlag
    ) -> TensorWithNormalisationFlag:

        if coordinates.normalised:
            coordinates_real_units = self.normaliser_x.inverse_transform(coordinates.tensor)
        else:
            coordinates_real_units = coordinates.tensor

        return TensorWithNormalisationFlag(
            tensor=coordinates_real_units,
            normalised=False
        )

    def print_status(self):
        raise NotImplementedError

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
        raise NotImplementedError("Aster, come in here with a debugger and confirm the shape >>:)")
        return self.evaluated_coordinates.shape[1]

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
    def pareto_optimal_points(self):
        raise NotImplementedError

    @property
    def evaluated_coordinates(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:
            coordinates = self.evaluated_coordinates_normalised
        else:
            coordinates = self.evaluated_coordinates_real_units

        return TensorWithNormalisationFlag(
            tensor=coordinates,
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
    def evaluated_coordinates_normalised(self) -> TensorWithNormalisationFlag:
        return self.normaliser_coordinates.transform(self.evaluated_coordinates_real_units)

    @cached_property
    def evaluated_values_normalised(self) -> TensorWithNormalisationFlag:
        return self.normaliser_values.transform(self.evaluated_values_real_units)

    @cached_property
    def bounds_normalised(self):
        return self.normaliser_coordinates.transform(self.bounds_real_units)

    @cached_property
    def initial_points_normalised(self):
        return self.normaliser_coordinates.transform(self.initial_points_real_units)
