import functools
from copy import deepcopy
from inspect import get_annotations
from typing import Callable, Optional, Self, Union, Unpack

import torch

torch.set_default_dtype(torch.float64)

from veropt.optimiser.initial_points import generate_initial_points
from veropt.optimiser.normalisation import Normaliser, get_normaliser_class
from veropt.optimiser.objective import (
    CallableObjective, InterfaceObjective, Objective, ObjectiveKind, determine_objective_type
)
from veropt.optimiser.optimiser_utility import (
    BestPoints, OptimisationMode,
    OptimiserSettings, OptimiserSettingsInputDict, ParetoOptimalPoints, SuggestedPoints,
    format_input_from_objective,
    format_output_for_objective, get_best_points, get_pareto_optimal_points,
    list_with_floats_to_string
)
from veropt.optimiser.prediction import Predictor
from veropt.optimiser.utility import (
    DataShape, TensorWithNormalisationFlag, check_variable_and_objective_shapes,
    enforce_amount_of_positional_arguments, unpack_flagged_variables_objectives_from_kwargs
)
from veropt.optimiser.saver_loader_utility import SavableClass, rehydrate_object


# TODO: Fix torch/np conflict over float size
#   - Put the torch default setting into this file, should we have it more places?
class BayesianOptimiser(SavableClass):

    def __init__(
            self,
            objective: Union[CallableObjective, InterfaceObjective],
            predictor: Predictor,
            normaliser_class: type[Normaliser],
            normaliser_variables: Optional[Normaliser],
            normaliser_objectives: Optional[Normaliser],
            settings: OptimiserSettings,
            initial_points_real_units: torch.Tensor,
            suggested_points: Optional[SuggestedPoints],
            suggested_points_history: list[SuggestedPoints],
            evaluated_variables_real_units: torch.Tensor,
            evaluated_objectives_real_units: torch.Tensor
    ):
        self.objective = objective
        self.n_objectives = objective.n_objectives
        self.objective_type = determine_objective_type(
            objective=objective
        )
        self._bounds_real_units = objective.bounds

        self.predictor = predictor

        self.settings = settings

        self._initial_points_real_units = initial_points_real_units

        self._evaluated_variables_real_units = evaluated_variables_real_units
        self._evaluated_objectives_real_units = evaluated_objectives_real_units

        self.suggested_points = suggested_points
        self.suggested_points_history = suggested_points_history

        self.normaliser_class = normaliser_class
        self._normaliser_variables = normaliser_variables
        self._normaliser_objectives = normaliser_objectives

        self.bounds_normalised: Optional[torch.Tensor] = None
        self.initial_points_normalised: Optional[torch.Tensor] = None
        self.evaluated_variables_normalised: Optional[torch.Tensor] = None
        self.evaluated_objectives_normalised: Optional[torch.Tensor] = None

        if self._normaliser_variables is not None and self._normaliser_objectives is not None:
            self._update_normalised_values()

        self._verify_set_up()
        self._set_up_settings()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"points: {self.n_points_evaluated} of {self.n_initial_points + self.n_bayesian_points}, "
            f"mode: {self.optimisation_mode.name}, "
            f"normalised: {'yes' if self.normalisers_have_been_initialised else 'no'}, "
            f"model trained: {'yes' if self.model_has_been_trained else 'no'}, "
            f"best objective values: {list_with_floats_to_string(self.get_best_points()['objectives'].tolist())}"
            f")"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"points: {self.n_points_evaluated} of {self.n_initial_points + self.n_bayesian_points}, \n"
            f"mode: {self.optimisation_mode.name}, \n"
            f"normalised: {'yes' if self.normalisers_have_been_initialised else 'no'}, \n"
            f"model trained: {'yes' if self.model_has_been_trained else 'no'}, \n"
            f"best objective values: {list_with_floats_to_string(self.get_best_points()['objectives'].tolist())}\n"
            f"{str(self.predictor)}"
            f")"
        )

    @classmethod
    def from_the_beginning(
            cls,
            n_initial_points: int,
            n_bayesian_points: int,
            n_evaluations_per_step: int,
            objective: Union[CallableObjective, InterfaceObjective],
            predictor: Predictor,
            normaliser_class: type[Normaliser],
            **kwargs: Unpack[OptimiserSettingsInputDict]
    ) -> 'BayesianOptimiser':

        objective = objective
        n_objectives = objective.n_objectives

        predictor = predictor
        normaliser_class = normaliser_class

        normaliser_variables = None
        normaliser_objectives = None

        # TODO: Move this assert somewhere else?
        # TODO: Write error message for this assert
        for key in kwargs.keys():
            assert key in get_annotations(OptimiserSettingsInputDict).keys()

        settings = OptimiserSettings(
            n_initial_points=n_initial_points,
            n_bayesian_points=n_bayesian_points,
            n_evaluations_per_step=n_evaluations_per_step,
            n_objectives=n_objectives,
            **kwargs
        )

        initial_points_real_units = generate_initial_points(
            initial_points_generator=settings.initial_points_generator,
            bounds=objective.bounds,
            n_initial_points=n_initial_points,
            n_variables=objective.n_variables
        )

        evaluated_variables_real_units = torch.tensor([])
        evaluated_objectives_real_units = torch.tensor([])

        suggested_points = None
        suggested_points_history: list[SuggestedPoints] = []

        return cls(
            objective=objective,
            predictor=predictor,
            normaliser_class=normaliser_class,
            settings=settings,
            initial_points_real_units=initial_points_real_units,
            suggested_points=suggested_points,
            suggested_points_history=suggested_points_history,
            normaliser_variables=normaliser_variables,
            normaliser_objectives=normaliser_objectives,
            evaluated_variables_real_units=evaluated_variables_real_units,
            evaluated_objectives_real_units=evaluated_objectives_real_units
        )

    @classmethod
    def from_saved_state(
            cls,
            saved_state: dict
    ) -> Self:

        objective = rehydrate_object(
            superclass=Objective,
            name=saved_state['objective']['name'],
            saved_state=saved_state['objective']['state']
        )

        assert issubclass(type(objective), CallableObjective) or issubclass(type(objective), InterfaceObjective)

        predictor = rehydrate_object(
            superclass=Predictor,
            name=saved_state['predictor']['name'],
            saved_state=saved_state['predictor']['state'],
        )

        if saved_state['normaliser_variables'] is None:
            normaliser_variables = None
        else:
            normaliser_variables = rehydrate_object(
                superclass=Normaliser,
                name=saved_state['normaliser_variables']['name'],
                saved_state=saved_state['normaliser_variables']['state']
            )

        if saved_state['normaliser_objectives'] is None:
            normaliser_objectives = None

        else:
            normaliser_objectives = rehydrate_object(
                superclass=Normaliser,
                name=saved_state['normaliser_objectives']['name'],
                saved_state=saved_state['normaliser_objectives']['state']
            )

        normaliser_class_name = saved_state['normaliser_class']
        normaliser_class = get_normaliser_class(
            normaliser_choice=normaliser_class_name
        )

        if normaliser_objectives is not None and normaliser_variables is not None:
            assert type(normaliser_objectives) is normaliser_class, (
                "Normalisers for variables and objectives must be of the same class."
            )

        initial_points_real_units = TensorWithNormalisationFlag(
            tensor=torch.tensor(saved_state['initial_points']['values']),
            normalised=saved_state['initial_points']['normalised'],
        )

        assert initial_points_real_units.normalised is False

        if len(saved_state['suggested_points']) == 0:
            suggested_points = None
        else:
            suggested_points = SuggestedPoints.from_saved_state(
                saved_state=saved_state['suggested_points']
            )

        suggested_points_history = [
            SuggestedPoints.from_saved_state(suggested_point_state)
            for suggested_point_state in saved_state['suggested_points_history']
        ]

        evaluated_variable_values = TensorWithNormalisationFlag(
            tensor=torch.tensor(saved_state['evaluated_variables']['values']),
            normalised=saved_state['evaluated_variables']['normalised']
        )

        assert evaluated_variable_values.normalised is False

        evaluated_objective_values = TensorWithNormalisationFlag(
            tensor=torch.tensor(saved_state['evaluated_objectives']['values']),
            normalised=saved_state['evaluated_objectives']['normalised']
        )

        assert evaluated_objective_values.normalised is False

        settings = OptimiserSettings.from_saved_state(
            saved_state['settings']
        )

        optimiser = cls(
            objective=objective,  # type: ignore[arg-type]  # this is checked above
            predictor=predictor,
            normaliser_class=normaliser_class,
            normaliser_variables=normaliser_variables,
            normaliser_objectives=normaliser_objectives,
            settings=settings,
            initial_points_real_units=initial_points_real_units.tensor,
            suggested_points=suggested_points,
            suggested_points_history=suggested_points_history,
            evaluated_variables_real_units=evaluated_variable_values.tensor,
            evaluated_objectives_real_units=evaluated_objective_values.tensor
        )

        if optimiser.model_has_been_trained:
            optimiser._update_predictor()

        return optimiser

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

            assert isinstance(args[0], BayesianOptimiser)
            self: BayesianOptimiser = args[0]

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

    def gather_dicts_to_save(self) -> dict:

        if self.normalisers_have_been_initialised:
            normaliser_variables_dict = self._normaliser_variables.gather_dicts_to_save()  # type: ignore[union-attr]
            normaliser_objectives_dict = self._normaliser_objectives.gather_dicts_to_save()  # type: ignore[union-attr]

        else:
            normaliser_variables_dict = None
            normaliser_objectives_dict = None

        return {
            'optimiser': {
                'objective': self.objective.gather_dicts_to_save(),
                'predictor': self.predictor.gather_dicts_to_save(),
                'initial_points': {
                    'values': self.initial_points_real_units,
                    'normalised': False
                },
                'normaliser_class': self.normaliser_class.name,
                'normaliser_variables': normaliser_variables_dict,
                'normaliser_objectives': normaliser_objectives_dict,
                'evaluated_variables': {
                    'values': self.evaluated_variables_real_units,
                    'normalised': False,
                },
                'evaluated_objectives': {
                    'values': self.evaluated_objectives_real_units,
                    'normalised': False,
                },
                'suggested_points': self.suggested_points.gather_dicts_to_save() if self.suggested_points else {},
                'suggested_points_history': [
                    suggested_point.gather_dicts_to_save() for suggested_point in self.suggested_points_history
                ],
                'settings': self.settings.gather_dicts_to_save()
            }
        }

    def run_optimisation_step(self) -> None:

        if self.objective_type == ObjectiveKind.callable:

            self.suggest_candidates()

            new_variables, new_values = self._evaluate_points()

            self._add_new_points(
                variable_values_flagged=new_variables,
                objective_values_flagged=new_values
            )

        elif self.objective_type == ObjectiveKind.interface:

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
            generated_with_mode=deepcopy(self.optimisation_mode.name),
            normalised=deepcopy(self.return_normalised_data)
        )

    def get_best_points(self) -> BestPoints:

        best_point = get_best_points(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor,
            weights=self.settings.objective_weights
        )

        assert best_point is not None, "Failed to get best point"

        return best_point

    def get_pareto_optimal_points(self) -> ParetoOptimalPoints:

        pareto_optimal_points = get_pareto_optimal_points(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor,
            weights=self.settings.objective_weights
        )

        return pareto_optimal_points

    def _evaluate_points(self) -> tuple[TensorWithNormalisationFlag, TensorWithNormalisationFlag]:

        assert self.objective_type == ObjectiveKind.callable, (
            f"The objective must be an {CallableObjective.__name__} to be evaluated during optimisation."
        )

        assert self.suggested_points is not None, "Suggested points must be created before using this function"

        new_variables_real_units = self._unnormalise_variables(
            variable_values_flagged=self.suggested_points.variable_values_flagged
        )

        objective_function_values = self.objective(new_variables_real_units.tensor)  # type: ignore[operator]

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
            *,
            variable_values_flagged: TensorWithNormalisationFlag,
            objective_values_flagged: TensorWithNormalisationFlag
    ) -> None:

        assert variable_values_flagged.normalised is False
        assert objective_values_flagged.normalised is False

        if len(variable_values_flagged.tensor) == 0 and len(objective_values_flagged.tensor) == 0:
            pass

        else:

            # TODO: Write good error message
            #   - Could also move this check somewhere...?
            #   - Then again, maybe we want a more flexible way to handle this in the future...?

            # TODO: Remove check or make setting to turn off
            #   - Consider if not checking this can make errors
            #   - Current step might be the main issue?
            assert variable_values_flagged.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step
            assert objective_values_flagged.tensor.shape[DataShape.index_points] == self.n_evaluations_per_step

            if self.n_points_evaluated == 0:

                self.evaluated_variables_real_units = variable_values_flagged.tensor.detach()
                self.evaluated_objectives_real_units = objective_values_flagged.tensor.detach()

            else:

                self.evaluated_variables_real_units = torch.cat(
                    tensors=[self.evaluated_variables_real_units, variable_values_flagged.tensor.detach()],
                    dim=DataShape.index_points
                )
                self.evaluated_objectives_real_units = torch.cat(
                    tensors=[self.evaluated_objectives_real_units, objective_values_flagged.tensor.detach()],
                    dim=DataShape.index_points
                )

            if self.model_has_been_trained:

                if self.settings.normalise:

                    if self.settings.renormalise_each_step:

                        self._fit_normaliser()

                self._update_predictor()

            elif self.n_points_evaluated > self.settings.n_points_before_fitting:

                if self.settings.normalise:

                    self._fit_normaliser()

                self._update_predictor()

            if self.settings.verbose:
                self._print_status()

    def _load_latest_points(self) -> None:

        assert self.objective_type == ObjectiveKind.interface, (
            f"The objective must be an {InterfaceObjective.__name__} to load points."
        )

        new_variable_values, new_objective_values = self.objective.load_evaluated_points()  # type: ignore[union-attr]

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

        assert self.objective_type == ObjectiveKind.interface, (
            "The objective must be an 'InterfaceObjective' to save candidates."
        )

        assert self.suggested_points is not None, "Must have made suggestions before saving them."

        suggested_variables_real_units = self._unnormalise_variables(
            variable_values_flagged=self.suggested_points.variable_values_flagged
        )

        suggested_variables_dict = format_output_for_objective(
            suggested_variables=suggested_variables_real_units.tensor,
            variable_names=self.objective.variable_names
        )

        self.objective.save_candidates(  # type: ignore[union-attr]  # objective type is checked above
            suggested_variables=suggested_variables_dict
        )

        self._reset_suggested_points()

    def _verify_set_up(self) -> None:

        assert self.n_initial_points > 0, "The number of initial points should be greater than 0."

        assert self.n_initial_points % self.n_evaluations_per_step == 0, (
            "The amount of initial points is not divisable by the amount of points evaluated each step."
        )

        assert self.n_bayesian_points % self.n_evaluations_per_step == 0, (
            "The amount of bayesian points is not divisable by the amount of points evaluated each step."
        )

    def _set_up_settings(self) -> None:

        # TODO: Make nan policy work
        #   - Need to handle in model but also in e.g. get_best_points and the like
        # if self.settings.mask_nans:
        #     gpytorch.settings.observation_nan_policy._set_value('mask')

        pass

    def _reset_suggested_points(self) -> None:

        if self.suggested_points is None:
            raise RuntimeError()

        # TODO: Unnormalise suggested point when saving
        self.suggested_points_history.append(self.suggested_points.copy())
        self.suggested_points = None

    def _update_predictor(self) -> None:

        if self.settings.normalise:
            assert self.normalisers_have_been_initialised

        self.predictor.update_with_new_data(
            variable_values=self.evaluated_variable_values.tensor,
            objective_values=self.evaluated_objective_values.tensor
        )

        self.suggested_points = None

    def _fit_normaliser(self) -> None:

        self._normaliser_variables = self.normaliser_class.from_tensor(
            tensor=self.evaluated_variables_real_units
        )

        self._normaliser_objectives = self.normaliser_class.from_tensor(
            tensor=self.evaluated_objectives_real_units
        )

        self._update_normalised_values()

        if self.settings.verbose:

            best_value_string = list_with_floats_to_string(self.get_best_points()['objectives'].tolist())
            print(f"Normalisation has been completed. Best values changed to: {best_value_string} \n")

    def _update_normalised_values(self) -> None:

        assert self._normaliser_variables is not None, "Normaliser must be initiated to update normalised values"
        assert self._normaliser_objectives is not None, "Normaliser must be initiated to update normalised values"

        self.initial_points_normalised = self._normaliser_variables.transform(
            tensor=self.initial_points_real_units
        )

        self.bounds_normalised = self._normaliser_variables.transform(
            tensor=self.bounds_real_units
        )

        self.evaluated_variables_normalised = self._normaliser_variables.transform(
            tensor=self.evaluated_variables_real_units
        )

        self.evaluated_objectives_normalised = self._normaliser_objectives.transform(
            tensor=self.evaluated_objectives_real_units
        )

        self.predictor.update_bounds(
            new_bounds=self.bounds.tensor
        )

    @_check_input_dimensions
    def _unnormalise_variables(
            self,
            *,
            variable_values_flagged: TensorWithNormalisationFlag
    ) -> TensorWithNormalisationFlag:

        if variable_values_flagged.normalised:

            assert self._normaliser_variables is not None, "Normaliser must be initialised at this point"

            variables_real_units = self._normaliser_variables.inverse_transform(variable_values_flagged.tensor)

        else:
            variables_real_units = variable_values_flagged.tensor

        return TensorWithNormalisationFlag(
            tensor=variables_real_units,
            normalised=False
        )

    def _print_status(self) -> None:

        best_point = self.get_best_points()

        best_variables, best_values, _ = (
            best_point['variables'], best_point['objectives'], best_point['index']
        )

        best_values_string = list_with_floats_to_string(best_values.tolist())

        best_values_variables_string = list_with_floats_to_string(best_variables.tolist())

        newest_value_string = list_with_floats_to_string(self.evaluated_objective_values[-1, :].tensor.tolist())

        newest_variables_string = list_with_floats_to_string(self.evaluated_variable_values[-1, :].tensor.tolist())

        total_steps = (self.n_initial_points + self.n_bayesian_points) // self.n_evaluations_per_step

        status_string = (
            f"Optimisation running in {self.optimisation_mode.name} mode "
            f"at step {self.current_step} out of {total_steps} \n"
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

        return self.n_points_evaluated // self.n_evaluations_per_step

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
    def model_has_been_trained(self) -> bool:
        return self.predictor.check_if_model_is_trained()

    @property
    def normalisers_have_been_initialised(self) -> bool:

        if self._normaliser_variables is not None and self._normaliser_objectives is not None:
            return True
        else:
            return False

    @property
    def return_normalised_data(self) -> bool:
        if self.settings.normalise and self.normalisers_have_been_initialised:
            return True
        else:
            return False

    @property
    def evaluated_variable_values(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:

            assert self.evaluated_variables_normalised is not None, (
                "Normalised tensor 'evaluated_variables_normalised' has not been initiated"
            )

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

            assert self.evaluated_objectives_normalised is not None, (
                "Normalised tensor 'evaluated_objective_normalised' has not been initiated"
            )

            values = self.evaluated_objectives_normalised
        else:
            values = self.evaluated_objectives_real_units

        return TensorWithNormalisationFlag(
            tensor=values,
            normalised=self.return_normalised_data
        )

    @property
    def bounds(self) -> TensorWithNormalisationFlag:
        if self.return_normalised_data:

            assert self.bounds_normalised is not None, (
                "Normalised tensor 'bounds_normalised' has not been initiated"
            )

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

            assert self.initial_points_normalised is not None, (
                "Normalised tensor 'initial_points_normalised' has not been initiated"
            )

            initial_points = self.initial_points_normalised
        else:
            initial_points = self.initial_points_real_units

        return TensorWithNormalisationFlag(
            tensor=initial_points,
            normalised=self.return_normalised_data
        )

    @property
    def evaluated_variables_real_units(self) -> torch.Tensor:
        return self._evaluated_variables_real_units

    @evaluated_variables_real_units.setter
    def evaluated_variables_real_units(self, variable_values: torch.Tensor) -> None:

        self._evaluated_variables_real_units = variable_values

        if self._normaliser_variables is not None:

            self.evaluated_variables_normalised = self._normaliser_variables.transform(
                tensor=self._evaluated_variables_real_units
            )

    @property
    def evaluated_objectives_real_units(self) -> torch.Tensor:
        return self._evaluated_objectives_real_units

    @evaluated_objectives_real_units.setter
    def evaluated_objectives_real_units(self, objective_values: torch.Tensor) -> None:

        self._evaluated_objectives_real_units = objective_values

        if self._normaliser_objectives is not None:

            self.evaluated_objectives_normalised = self._normaliser_objectives.transform(
                tensor=self._evaluated_objectives_real_units
            )

    @property
    def bounds_real_units(self) -> torch.Tensor:
        return self._bounds_real_units

    @bounds_real_units.setter
    def bounds_real_units(self, bounds: torch.Tensor) -> None:
        self._bounds_real_units = bounds

        if self._normaliser_variables is not None:

            self.bounds_normalised = self._normaliser_variables.transform(
                tensor=self._bounds_real_units
            )

    @property
    def initial_points_real_units(self) -> torch.Tensor:
        return self._initial_points_real_units

    @initial_points_real_units.setter
    def initial_points_real_units(self, initial_points: torch.Tensor) -> None:
        self._initial_points_real_units = initial_points

        if self._normaliser_variables is not None:

            self.initial_points_normalised = self._normaliser_variables.transform(
                tensor=self._initial_points_real_units
            )
