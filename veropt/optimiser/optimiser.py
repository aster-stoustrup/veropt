import abc
from copy import deepcopy
from inspect import get_annotations
from typing import Optional, Union, Unpack

import torch

from veropt.optimiser.acquisition import AcquisitionFunction
from veropt.optimiser.model import SurrogateModel
from veropt.optimiser.normaliser import Normaliser
from veropt.optimiser.optimiser_utility import InitialPointsGenerationMode, ObjectiveType, OptimisationMode, \
    OptimiserSettings, OptimiserSettingsInputDict, SuggestedPoints, TensorWithNormalisationFlag


class Objective:
    def __init__(
            self,
            bounds: torch.Tensor,
            n_parameters: int,
            n_objectives: int,
            parameter_names: list[str] = None,
            objective_names: list[str] = None
    ):
        self.bounds = bounds
        self.n_parameters = n_parameters
        self.n_objectives = n_objectives

        if parameter_names is None:
            self.parameter_names = [f"Parameter {i}" for i in range(1, n_parameters + 1)]
        else:
            self.parameter_names = parameter_names

        if objective_names is None:
            self.objective_names = [f"Objective {i}" for i in range(1, n_objectives + 1)]
        else:
            self.objective_names = objective_names


class IntegratedObjective(Objective):

    @abc.abstractmethod
    def run(self, parameter_values):
        pass


class InterfaceObjective(Objective):

    @abc.abstractmethod
    def save_candidates(self):
        pass

    @abc.abstractmethod
    def load_evaluated_points(self):
        pass


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
            normaliser: Normaliser = None,
            **kwargs: Unpack[OptimiserSettingsInputDict]
    ):

        self.objective = objective

        # TODO: Consider making methods for these to keep init as clean as possible?
        if acquisition_function is None:
            raise NotImplementedError
        else:
            self.acquisition_function = acquisition_function

        if model is None:
            raise NotImplementedError
        else:
            self.model = model

        if normaliser is None:
            raise NotImplementedError
        else:
            self.normaliser = normaliser

        # TODO: Move this asser somewhere else?
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

        # TODO: Decide on consistent, nice naming for these two
        self.data_normalised = False
        self.model_is_fitted = False

        self.initial_points_normalised = None
        self.evaluated_coordinates_normalised = None
        self.evaluated_values_normalised = None
        self.bounds_normalised = None

        self.suggested_points: Optional[SuggestedPoints] = None
        self.suggested_points_history: list[Optional[SuggestedPoints]] = (
                [None] * (self.n_initial_points + self.n_bayesian_points)
        )

        self.objective_type = self.determine_objective_type()

        self.verify_set_up()

    def determine_objective_type(self) -> ObjectiveType:
        if issubclass(type(self.objective), IntegratedObjective):
            return ObjectiveType.integrated
        elif issubclass(type(self.objective), InterfaceObjective):
            return ObjectiveType.interface
        else:
            raise ValueError("The objective must be a subclass of either IntegratedObjective or InterfaceObjective.")

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

        if self.objective_type == ObjectiveType.integrated:

            self.suggest_candidates()

            new_coordinates, new_values = self.evaluate_points()

            self.add_new_points(new_coordinates, new_values)

        elif self.objective_type == ObjectiveType.interface:

            self.load_latest_points()

            self.suggest_candidates()

            self.save_candidates()

    def evaluate_points(self) -> (TensorWithNormalisationFlag, TensorWithNormalisationFlag):

        assert self.objective_type == ObjectiveType.integrated, (
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

            raise NotImplementedError

        if self.model_is_fitted:
            predicted_values = self.model(suggested_coordinates.tensor)
        else:
            predicted_values = None

        self.suggested_points = SuggestedPoints(
            coordinates=suggested_coordinates,
            predicted_values=predicted_values,
            generated_at_step=self.current_step
        )

    def add_new_points(
            self,
            new_coordinates: TensorWithNormalisationFlag,
            new_values: TensorWithNormalisationFlag
    ):
        raise NotImplementedError

    def reset_suggested_points(self):
        self.suggested_points_history[self.current_step - 1] = deepcopy(self.suggested_points)
        self.suggested_points = None

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

    @property
    def return_normalised_data(self) -> bool:
        if self.settings.normalise and self.data_normalised:
            return True
        else:
            return False

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

    @property
    def n_initial_points(self) -> int:
        return self.settings.n_initial_points

    @property
    def n_bayesian_points(self) -> int:
        return self.settings.n_bayesian_points

    @property
    def current_step(self):
        assert self.n_points_evaluated % self.n_evaluations_per_step == 0, (
            "Amount of points evaluated does not match step size."
        )
        return self.n_points_evaluated // self.n_evaluations_per_step + 1

    @property
    def n_evaluations_per_step(self) -> int:
        return self.settings.n_evaluations_per_step

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
