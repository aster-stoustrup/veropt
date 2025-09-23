from __future__ import annotations

from copy import deepcopy
from typing import Literal, Optional, Union

import numpy as np
import plotly.graph_objs as go
import torch
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from plotly import colors
from plotly.subplots import make_subplots

from veropt.graphical.visualisation_utility import (
    ModelPrediction, ModelPredictionContainer,
    opacity_for_multidimensional_points, get_continuous_colour
)
from veropt.optimiser.acquisition import AcquisitionFunction
from veropt.optimiser.acquisition_optimiser import (
    ProximityPunishAcquisitionFunction,
    ProximityPunishmentSequentialOptimiser
)
from veropt.optimiser.optimiser import BayesianOptimiser
# from veropt.utility import opacity_for_multidimensional_points
# ModelPrediction, ModelPredictionContainer
from veropt.optimiser.optimiser_utility import SuggestedPoints, get_best_points
from veropt.optimiser.prediction import BotorchPredictor
from veropt.optimiser.utility import DataShape


def plot_point_overview_from_optimiser(
        optimiser: BayesianOptimiser,
        points: Literal['all', 'bayes', 'suggested', 'best'] = 'all'
) -> None:

    n_objectives = optimiser.n_objectives

    shown_inds = None

    if points == 'all':
        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor

    elif points == 'bayes':

        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor

        shown_inds = np.arange(optimiser.n_initial_points, optimiser.n_points_evaluated).tolist()

    elif points == 'suggested':

        assert optimiser.suggested_points, "Must have active suggested points to choose this option"

        suggested_points = optimiser.suggested_points

        variable_values = suggested_points.variable_values

        assert suggested_points.predicted_objective_values is not None, (
            "Must have calculated predictions for the suggested points before calling this function to plot them."
            "(If the model is trained, the optimiser should do this automatically)."
        )

        objective_values = suggested_points.predicted_objective_values['mean']

    elif points == 'best':

        # TODO: Might be optimal to open all points but mark the best ones or make them visible or something

        best_indices = []

        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor

        best_points_general = get_best_points(
            variable_values=variable_values,
            objective_values=objective_values,
            weights=optimiser.settings.objective_weights
        )

        assert best_points_general is not None, "Failed to find best points"

        best_indices.append(best_points_general['index'])

        for objective_index in range(n_objectives):

            best_points_for_objective = get_best_points(
                variable_values=variable_values,
                objective_values=objective_values,
                weights=optimiser.settings.objective_weights,
                best_for_objecive_index=objective_index
            )

            assert best_points_for_objective is not None, f"Failed to find best points for objective {objective_index}"

            best_indices.append(
                best_points_for_objective['index']
            )

        shown_inds = np.unique(best_indices).tolist()  # type: ignore[assignment]  # checking below
        assert type(shown_inds) is list
        assert type(shown_inds[0]) is str

    else:
        raise ValueError

    objective_names = optimiser.objective.objective_names
    variable_names = optimiser.objective.variable_names

    plot_point_overview(
        variable_values=variable_values,
        objective_values=objective_values,
        objective_names=objective_names,
        variable_names=variable_names,
        shown_indices=shown_inds
    )


# TODO: Untangle all visualisation tools from god object and put them in here


# TODO: Add type hints
# TODO: Find better name, could also be used for evaluated points
#   - When we do this, also need to rename input names
def plot_point_overview(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        objective_names: list[str],
        variable_names: list[str],
        shown_indices: Optional[list[int]] = None
) -> None:
    # TODO: Maybe want a longer colour scale to avoid duplicate colours...?
    color_scale = colors.qualitative.T10
    color_scale = colors.convert_colors_to_same_type(color_scale, colortype="rgb")[0]
    n_colors = len(color_scale)

    # TODO: Cool hover shit?
    #   - Even without a dash app, we could add the "sum score" for each point on hover

    n_points = variable_values.shape[0]

    opacity_lines = 0.2

    figure = make_subplots(rows=2, cols=1)

    # TODO: Give the point numbers of all evaluated points (unless it's suggested points?)
    for point_no in range(n_points):

        if shown_indices is not None:
            if point_no not in shown_indices:
                args = {'visible': 'legendonly'}
            else:
                args = {}
        else:
            args = {}

        figure.add_trace(
            go.Scatter(
                x=variable_names,
                y=variable_values[point_no].detach().numpy(),
                name=f"Point no. {point_no}",  # This is currently out of the ones plotted, consider that
                line={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", {opacity_lines})"},
                marker={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + ", 1.0)"},
                mode='lines+markers',
                legendgroup=point_no,
                **args
            ),
            row=1,
            col=1
        )

        figure.add_trace(
            go.Scatter(
                x=objective_names,
                y=objective_values[point_no].detach().numpy(),
                line={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", {opacity_lines})"},
                marker={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + ", 1.0)"},
                name=f"Point no. {point_no}",
                mode='lines+markers',
                legendgroup=point_no,
                showlegend=False,
                **args
            ),
            row=2,
            col=1
        )

    figure.update_layout(
        # title={'text': "Plot Title"},
        # xaxis={'title': {'text': "Parameter Number"}},  # Maybe obvious and unnecessary?
        yaxis={'title': {'text': "Parameter Values"}},  # TODO: Add if they're normalised or not
        # TODO: Add if they're predicted or evaluated
        yaxis2={'title': {'text': "Objective Values"}},  # TODO: Add if they're normalised or not
    )

    if n_points < 7:
        figure.update_layout(hovermode="x")

    figure.show()


def plot_progression(
        objective_values: torch.Tensor,
        objective_names: list[str],
        n_initial_points: int
) -> go.Figure:

    n_evaluated_points = objective_values.shape[DataShape.index_points]
    n_objectives = objective_values.shape[DataShape.index_dimensions]

    colour_scale = colors.qualitative.Dark2
    if n_objectives > 1:
        colour_list = colors.sample_colorscale(
            colorscale=colour_scale,
            samplepoints=n_objectives
        )
    else:
        colour_list = [colour_scale[0]]

    figure = go.Figure()

    for objective_index in range(n_objectives):

        figure.add_trace(go.Scatter(
            x=np.arange(n_initial_points),
            y=objective_values.detach().numpy()[:n_initial_points, objective_index],
            name=f"Initial points, objective '{objective_names[objective_index]}'",
            mode='markers',
            marker={
                'symbol': 'diamond',
                'color': colour_list[objective_index],
            },
        ))

        figure.add_trace(go.Scatter(
            x=np.arange(n_initial_points, n_evaluated_points),
            y=objective_values.detach().numpy()[n_initial_points:, objective_index],
            name=f"Bayesian points, objective '{objective_names[objective_index]}'",
            mode='markers',
            marker={
                'color': colour_list[objective_index],
            }
        ))

    figure.update_layout(
        xaxis={
            'title': {
                'text': "Evaluated points",
            }},
        yaxis={
            'title': {
                'text': "Objective values",
            }},

    )

    return figure


def plot_progression_from_optimiser(
        optimiser: BayesianOptimiser,
        normalised: Optional[bool] = None,
        return_figure: bool = False
) -> Union[None, go.Figure]:

    if normalised is None:
        if optimiser.settings.normalise and optimiser.normalisers_have_been_initialised:
            normalised = True
        else:
            normalised = False

    if normalised:
        objective_values = optimiser.evaluated_objectives_normalised
        assert objective_values is not None
    else:
        objective_values = optimiser.evaluated_objectives_real_units

    figure = plot_progression(
        objective_values=objective_values,
        objective_names=optimiser.objective.objective_names,
        n_initial_points=optimiser.n_initial_points,
    )

    if return_figure:
        return figure
    else:
        figure.show()
        return None


def plot_pareto_front_grid_from_optimiser(
        optimiser: BayesianOptimiser
) -> None:
    variable_values = optimiser.evaluated_variable_values.tensor
    pareto_optimal_objectives = optimiser.get_pareto_optimal_points()['objectives']

    objective_names = optimiser.objective.objective_names

    plot_pareto_front_grid(
        variable_values=variable_values,
        objective_names=objective_names,
        dominating_objective_values=pareto_optimal_objectives,
        suggested_points=optimiser.suggested_points,
    )


def plot_pareto_front_grid(
        variable_values: torch.Tensor,
        objective_names: list[str],
        dominating_objective_values: torch.Tensor,
        suggested_points: Optional[SuggestedPoints] = None,
        return_figure: bool = False
) -> Union[go.Figure, None]:

    n_objectives = len(objective_names)

    figure = make_subplots(
        rows=n_objectives - 1,
        cols=n_objectives - 1
    )

    for objective_index_x in range(n_objectives - 1):
        for objective_index_y in range(1, n_objectives):

            row = objective_index_y
            col = objective_index_x + 1

            if not objective_index_x == objective_index_y:
                figure = _add_pareto_traces_2d(
                    figure=figure,
                    objective_values=variable_values,
                    objective_index_x=objective_index_x,
                    objective_index_y=objective_index_y,
                    dominating_objective_values=dominating_objective_values,
                    suggested_points=suggested_points,
                    row=row,
                    col=col
                )

            if col == 1:
                figure.update_yaxes(title_text=objective_names[objective_index_y], row=row, col=col)

            if row == n_objectives - 1:
                figure.update_xaxes(title_text=objective_names[objective_index_x], row=row, col=col)

    if return_figure:

        return figure
    else:

        figure.show()
        return None


def _add_pareto_traces_2d(
        figure: go.Figure,
        objective_values: torch.Tensor,
        objective_index_x: int,
        objective_index_y: int,
        dominating_objective_values: torch.Tensor,
        suggested_points: Optional[SuggestedPoints] = None,
        row: Optional[int] = None,
        col: Optional[int] = None
) -> go.Figure:

    if row is None and col is None:
        row_col_info = {}

    else:
        row_col_info = {
            'row': row,
            'col': col
        }

    color_scale = colors.qualitative.Plotly
    color_evaluated_points = color_scale[0]

    figure.add_trace(
        go.Scatter(
            x=objective_values[:, objective_index_x],
            y=objective_values[:, objective_index_y],
            mode='markers',
            name='Evaluated points',
            marker={'color': color_evaluated_points},
        ),
        **row_col_info
    )

    figure.add_trace(
        go.Scatter(
            x=dominating_objective_values[:, objective_index_x],
            y=dominating_objective_values[:, objective_index_y],
            mode='markers',
            marker={'color': 'black'},
            name='Dominating evaluated points'
        ),
        **row_col_info
    )

    if suggested_points is not None:

        suggested_point_color = 'rgb(139, 0, 0)'

        for suggested_point_no, point in enumerate(suggested_points):

            prediction = point.predicted_objective_values

            assert prediction is not None, (
                "Must have calculated predictions for the suggested points before calling this function to plot them."
                "(If the model is trained, the optimiser should do this automatically)."
            )

            upper_diff = prediction['upper'] - prediction['mean']
            lower_diff = prediction['mean'] - prediction['lower']

            figure.add_trace(
                go.Scatter(
                    x=prediction['mean'][objective_index_x].detach().numpy(),
                    y=prediction['mean'][objective_index_y].detach().numpy(),
                    error_x={
                        'type': 'data',
                        'symmetric': False,
                        'array': upper_diff[objective_index_x].detach().numpy(),
                        'arrayminus': lower_diff[objective_index_x].detach().numpy(),
                        'color': suggested_point_color
                    },
                    error_y={
                        'type': 'data',
                        'symmetric': False,
                        'array': upper_diff[objective_index_y].detach().numpy(),
                        'arrayminus': lower_diff[objective_index_y].detach().numpy(),
                        'color': suggested_point_color
                    },
                    mode='markers',
                    marker={'color': suggested_point_color},
                    name='Suggested point',
                ),
                **row_col_info
            )

    return figure


def plot_pareto_front(
        objective_values: torch.Tensor,
        dominating_objective_values: torch.Tensor,
        plotted_objective_indices: list[int],
        suggested_points: Optional[SuggestedPoints] = None,
        return_figure: bool = False
) -> Union[go.Figure, None]:

    if len(plotted_objective_indices) == 2:

        obj_ind_x = plotted_objective_indices[0]
        obj_ind_y = plotted_objective_indices[1]

        figure = go.Figure()

        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=obj_ind_x,
            objective_index_y=obj_ind_y,
            dominating_objective_values=dominating_objective_values,
            suggested_points=suggested_points
        )

    elif len(plotted_objective_indices) == 3:

        # TODO: Add suggested points
        # TODO: Add dominating points

        plotted_obj_vals = objective_values[:, plotted_objective_indices]

        figure = go.Figure(data=[go.Scatter3d(
            x=plotted_obj_vals[:, plotted_objective_indices[0]],
            y=plotted_obj_vals[:, plotted_objective_indices[1]],
            z=plotted_obj_vals[:, plotted_objective_indices[2]],
            mode='markers'
        )])

    else:
        raise ValueError(f"Can plot pareto front of either 2 or 3 objectives, got {len(plotted_objective_indices)}")

    if return_figure:
        return figure

    else:
        figure.show()
        return None


def plot_pareto_front_from_optimiser(
        optimiser: BayesianOptimiser,
        plotted_objective_indices: list[int]
) -> None:
    objective_values = optimiser.evaluated_objective_values.tensor
    pareto_optimal_objectives = optimiser.get_pareto_optimal_points()['objectives']

    plot_pareto_front(
        objective_values=objective_values,
        dominating_objective_values=pareto_optimal_objectives,
        plotted_objective_indices=plotted_objective_indices,
        suggested_points=optimiser.suggested_points
    )


# TODO: Move somewhere nice
def _calculate_proximity_punished_acquisition_values(
        optimiser: BayesianOptimiser,
        evaluated_point: torch.Tensor,
        variable_index: int,
        variable_array: np.ndarray,
        acquisition_function: AcquisitionFunction,
        suggested_points_variables: torch.Tensor
) -> list[torch.Tensor]:

    n_suggested_points = suggested_points_variables.shape[DataShape.index_points]

    assert type(optimiser.predictor) is BotorchPredictor
    predictor: BotorchPredictor = optimiser.predictor

    assert type(predictor.acquisition_optimiser) is ProximityPunishmentSequentialOptimiser
    acquisition_optimiser: ProximityPunishmentSequentialOptimiser = predictor.acquisition_optimiser

    assert acquisition_optimiser.scaling is not None, (
        "Must have calculated scaling in proximity punishment acquisition optimiser before calling this"
    )

    punishing_acquisition_function = ProximityPunishAcquisitionFunction(
        original_acquisition_function=acquisition_function,
        other_points=[],
        scaling=acquisition_optimiser.scaling,
        alpha=acquisition_optimiser.settings.alpha,
        omega=acquisition_optimiser.settings.omega
    )

    modified_acquisition_values: list[torch.Tensor] = []

    full_variable_array = evaluated_point.repeat(len(variable_array), 1)
    full_variable_array[:, variable_index] = torch.tensor(variable_array)

    suggested_points_variables_list = [suggested_points_variables[i, :] for i in range(n_suggested_points)]

    for last_included_point_no in range(n_suggested_points - 1):

        punishing_acquisition_function.update_points(
            new_points=suggested_points_variables_list[0:last_included_point_no + 1]
        )

        modified_acquisition_values.append(punishing_acquisition_function(
            variable_values=full_variable_array,
        ))

    return modified_acquisition_values


def choose_plot_point(
        optimiser: BayesianOptimiser
) -> tuple[torch.Tensor, str]:

    if optimiser.suggested_points is None:
        # TODO: Use some general method instead of hard-coding this here
        max_ind = optimiser.get_best_points()['index']
        eval_point = deepcopy(optimiser.evaluated_variable_values[max_ind:max_ind + 1].tensor)
        point_description = f"at the point with the highest known value (point no. {max_ind})"
    else:
        suggested_point_ind = 0  # In the future, might want the best one
        eval_point = deepcopy(optimiser.suggested_points.variable_values[suggested_point_ind:suggested_point_ind + 1])
        point_description = "at the first suggested step"

    return eval_point, point_description


def fill_model_prediction_from_optimiser(
        optimiser: BayesianOptimiser,
        variable_index: int,
        evaluated_point: Optional[torch.Tensor],
        n: int = 200
) -> ModelPrediction:

    if evaluated_point is None:

        evaluated_point, title = choose_plot_point(
            optimiser=optimiser
        )

    else:
        title = ''

    variable_array = np.linspace(
        start=optimiser.bounds[0, variable_index].tensor,
        stop=optimiser.bounds[1, variable_index].tensor,
        num=n
    )

    all_variables_array = evaluated_point.repeat(len(variable_array), 1)
    all_variables_array[:, variable_index] = torch.tensor(variable_array)  # TODO: Figure out variable_array type issue

    return ModelPrediction(
        variable_index=variable_index,
        point=evaluated_point,
        title=title,
        variable_array=variable_array,
        predicted_objective_values=optimiser.predictor.predict_values(
            variable_values=all_variables_array
        ),
        acquisition_values=optimiser.predictor.get_acquisition_values(
            variable_values=all_variables_array
        ),
        samples=None
    )


def plot_prediction_grid_from_optimiser(
        optimiser: BayesianOptimiser,
        return_figure: bool = False,
        model_prediction_container: Optional[ModelPredictionContainer] = None,
        evaluated_point: Optional[torch.Tensor] = None
) -> Union[go.Figure, None]:

    variable_values = optimiser.evaluated_variable_values.tensor
    objective_values = optimiser.evaluated_objective_values.tensor
    objective_names = optimiser.objective.objective_names
    variable_names = optimiser.objective.variable_names

    n_variables = variable_values.shape[DataShape.index_dimensions]

    if model_prediction_container is None:
        model_prediction_container = ModelPredictionContainer()

    if evaluated_point is None:
        # I guess there's a non-caught case where no point was chosen but the auto-selected point is already calculated
        calculate_new_predictions = True

    elif evaluated_point in model_prediction_container:
        calculate_new_predictions = False

    elif evaluated_point not in model_prediction_container:
        calculate_new_predictions = True

    else:
        raise RuntimeError("Unexpected error.")

    if calculate_new_predictions:

        for var_ind in range(n_variables):

            calculated_prediction = fill_model_prediction_from_optimiser(
                optimiser=optimiser,
                variable_index=var_ind,
                evaluated_point=evaluated_point,
            )

            if optimiser.suggested_points:

                if type(optimiser.predictor) is BotorchPredictor:

                    if type(optimiser.predictor.acquisition_optimiser) is ProximityPunishmentSequentialOptimiser:

                        punished_acquisition_values = _calculate_proximity_punished_acquisition_values(
                            optimiser=optimiser,
                            evaluated_point=calculated_prediction.point,
                            variable_index=var_ind,
                            variable_array=calculated_prediction.variable_array,
                            acquisition_function=optimiser.predictor.acquisition_function,
                            suggested_points_variables=optimiser.suggested_points.variable_values
                        )

                        calculated_prediction.add_modified_acquisition_values(
                            modified_acquisition_values=punished_acquisition_values
                        )

            model_prediction_container.add_data(
                model_prediction=calculated_prediction
            )

    if evaluated_point is None:
        evaluated_point = calculated_prediction.point

    suggested_points = optimiser.suggested_points

    figure = plot_prediction_grid(
        model_prediction_container=model_prediction_container,
        evaluated_point=evaluated_point,
        variable_values=variable_values,
        objective_values=objective_values,
        objective_names=objective_names,
        variable_names=variable_names,
        suggested_points=suggested_points
    )

    if return_figure:

        return figure

    else:

        figure.show()
        return None


def _add_model_traces(
        figure: go.Figure,
        model_prediction: ModelPrediction,
        row_no: int,
        col_no: int,
        objective_index: int,
        legend_group: str
) -> None:

    predicted_values_mean = model_prediction.predicted_values_mean[:, objective_index]
    predicted_values_lower = model_prediction.predicted_values_lower[:, objective_index]
    predicted_values_upper = model_prediction.predicted_values_upper[:, objective_index]

    figure.add_trace(
        go.Scatter(
            x=model_prediction.variable_array,
            y=predicted_values_upper.detach().numpy(),
            line={'width': 0.0, 'color': 'rgba(156, 156, 156, 0.4)'},
            name='Upper bound prediction',
            legendgroup=legend_group,
            showlegend=False
        ),
        row=row_no, col=col_no
    )

    figure.add_trace(
        go.Scatter(
            x=model_prediction.variable_array,
            y=predicted_values_lower.detach().numpy(),
            fill='tonexty',  # This fills between this and the line above
            line={'width': 0.0, 'color': 'rgba(156, 156, 156, 0.4)'},
            name='Lower bound prediction',
            legendgroup=legend_group,
            showlegend=False,
        ),
        row=row_no, col=col_no
    )

    figure.add_trace(
        go.Scatter(
            x=model_prediction.variable_array,
            y=predicted_values_mean.detach().numpy(),
            line={'color': 'black'},
            name='Mean prediction',
            legendgroup=legend_group,
            showlegend=True if (row_no == 1 and col_no == 1) else False
        ),
        row=row_no, col=col_no
    )


def plot_prediction_grid(
        model_prediction_container: ModelPredictionContainer,
        evaluated_point: torch.Tensor,
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        objective_names: list[str],
        variable_names: list[str],
        suggested_points: Optional[SuggestedPoints] = None
) -> go.Figure:

    # TODO: Add option to plot subset of all these
    #   - Could be from var/obj start_ind to var/obj end_ind
    #   - Could be lists of vars and objs
    #   - Could be single var or single obj
    #   - Could be mix of these

    n_evaluated_points = variable_values.shape[DataShape.index_points]

    if suggested_points:
        n_suggested_points = len(suggested_points)

    n_variables = variable_values.shape[1]
    n_objectives = len(objective_names)

    colour_scale = colors.get_colorscale('matter')
    colour_scale_suggested_points = colors.get_colorscale('Emrld')
    # colour_list = colors.sample_colorscale(
    #     colorscale=colour_scale,
    #     samplepoints=n_evaluated_points,
    #     low=0.0,
    #     high=1.0,
    #     colortype='rgb'
    # )

    figure = make_subplots(
        rows=n_objectives,
        cols=n_variables
    )

    for variable_index in range(n_variables):

        model_prediction = model_prediction_container(
            variable_index=variable_index,
            point=evaluated_point
        )

        if suggested_points:
            joint_points = torch.concat([
                variable_values,
                suggested_points.variable_values
            ])
        else:
            joint_points = variable_values

        joint_opacity_list, joint_distance_list = opacity_for_multidimensional_points(
            variable_index=variable_index,
            n_variables=n_variables,
            variable_values=joint_points,
            evaluated_point=evaluated_point,
            alpha_min=0.4,
            alpha_max=1.0
        )

        distance_list = joint_distance_list[:n_evaluated_points]
        suggested_point_distance_list = joint_distance_list[n_evaluated_points:]

        marker_type_list = ['circle'] * n_evaluated_points
        marker_size_list = [8] * n_evaluated_points

        evaluated_point_ind = np.where(joint_distance_list == 0.0)[0][0]

        if evaluated_point_ind < n_evaluated_points:
            marker_type_list[evaluated_point_ind] = 'x'
            marker_size_list[evaluated_point_ind] = 14

        if evaluated_point_ind >= n_evaluated_points:
            evaluated_suggested_point_ind = evaluated_point_ind - n_evaluated_points

        else:
            evaluated_suggested_point_ind = None

        colour_list = [get_continuous_colour(colour_scale, float(1 - distance)) for distance in distance_list]

        colour_list_w_opacity = [
            "rgba(" + colour_list[point_no][4:-1] + f", {joint_opacity_list[point_no]})"
            for point_no in range(n_evaluated_points)
        ]

        if suggested_points:

            colour_list_suggested_points = [
                get_continuous_colour(colour_scale_suggested_points, float(1 - distance))
                for distance in suggested_point_distance_list
            ]

            colour_list_suggested_points_w_opacity = [
                (
                    f"rgba("
                    f"{colour_list_suggested_points[point_no][4:-1]}, "
                    f"{joint_opacity_list[n_evaluated_points + point_no]}"
                    f")"
                )
                for point_no in range(n_suggested_points)
            ]

        for objective_index in range(n_objectives):

            # Placing these backwards to make the "y axes" of subplots go positive upwards
            row_no = n_objectives - objective_index
            col_no = variable_index + 1

            # Quick scaling as long as we're just jamming it into this plot
            # acq_func_scaling = np.abs(model_pred_data.acq_fun_vals).max() * 0.5
            acq_func_scaling = 1.0

            _add_model_traces(
                figure=figure,
                model_prediction=model_prediction,
                row_no=row_no,
                col_no=col_no,
                objective_index=objective_index,
                legend_group='model'
            )

            # TODO: Make acq func colours nicer
            acquisition_function_colour = 'grey'

            figure.add_trace(
                go.Scatter(
                    x=model_prediction.variable_array,
                    y=model_prediction.acquisition_values / acq_func_scaling,
                    line={'color': acquisition_function_colour},
                    name='Acquisition function',
                    legendgroup='acq func',
                    showlegend=True if (row_no == 1 and col_no == 1) else False
                ),
                row=row_no, col=col_no
            )

            if model_prediction.modified_acquisition_values is not None:

                for punish_ind, punished_acq_fun_vals in enumerate(model_prediction.modified_acquisition_values):

                    figure.add_trace(
                        go.Scatter(
                            x=model_prediction.variable_array,
                            y=punished_acq_fun_vals.detach() / acq_func_scaling,
                            line={'color': acquisition_function_colour},
                            name=f'Acq. func., as seen by suggested point {punish_ind + 1}',
                            legendgroup=f'acq func {punish_ind}',
                            showlegend=True if (row_no == 1 and col_no == 1) else False
                        ),
                        row=row_no, col=col_no
                    )

            figure.add_trace(
                go.Scatter(
                    x=variable_values[:, variable_index],
                    y=objective_values[:, objective_index],
                    mode='markers',
                    marker={
                        'color': colour_list_w_opacity,
                        'size': marker_size_list,
                        'line': {
                            'width': 0.0
                        },
                        'opacity': 1.0
                    },
                    marker_symbol=marker_type_list,
                    name='Evaluated points',
                    showlegend=True if (row_no == 1 and col_no == 1) else False,
                    legendgroup='Evaluated points',
                    customdata=np.dstack([list(range(n_evaluated_points)), distance_list])[0],
                    hovertemplate="Param. value: %{x:.3f} <br>"
                                  "Obj. func. value: %{y:.3f} <br>"
                                  "Point number: %{customdata[0]:.0f} <br>"
                                  "Distance to current plane: %{customdata[1]:.3f}"
                ),
                row=row_no,
                col=col_no
            )

            if suggested_points:
                for suggested_point_no, point in enumerate(suggested_points):

                    if suggested_point_no == evaluated_suggested_point_ind:
                        marker_style = 'x'
                        marker_size = 14  # TODO: Write these somewhere general
                    else:
                        marker_style = 'circle'
                        marker_size = 8

                    prediction = point.predicted_objective_values
                    assert prediction is not None, (
                        "Must have calculated predictions for the suggested points before calling this function to "
                        "plot them. (If the model is trained, the optimiser should do this automatically)."
                    )

                    show_legend = False
                    if variable_index == 0 and objective_index == 0 and suggested_point_no == 0:
                        show_legend = True

                    upper_diff = prediction['upper'] - prediction['mean']
                    lower_diff = prediction['mean'] - prediction['lower']

                    figure.add_trace(
                        go.Scatter(
                            x=point.variable_values[variable_index].detach().numpy(),
                            y=prediction['mean'][objective_index].detach().numpy(),
                            error_y={
                                'type': 'data',
                                'symmetric': False,
                                'array': upper_diff[objective_index].detach().numpy(),
                                'arrayminus': lower_diff[objective_index].detach().numpy(),
                                'color': colour_list_suggested_points_w_opacity[suggested_point_no]
                            },
                            mode='markers',
                            marker={
                                'color': colour_list_suggested_points_w_opacity[suggested_point_no],
                                'size': marker_size
                            },
                            marker_symbol=marker_style,
                            name='Suggested points',
                            legendgroup='Suggested points',
                            showlegend=show_legend,
                            customdata=np.dstack([
                                [suggested_point_no],
                                [suggested_point_distance_list[suggested_point_no]],
                                [upper_diff[objective_index].detach().numpy()],
                                [lower_diff[objective_index].detach().numpy()]
                            ])[0],
                            # TODO: Super sweet feature would be to check if upper and lower are equal and then do pm
                            hovertemplate="Param. value: %{x:.3f} <br>"
                                          "Obj. func. value: %{y:.3f}"
                                          " + %{customdata[2]:.3f} /"
                                          " - %{customdata[3]:.3f} <br>"
                                          "Suggested point number: %{customdata[0]:.0f} <br>"
                                          "Distance to current plane: %{customdata[1]:.3f}"
                        ),
                        row=row_no, col=col_no
                    )

            figure.update_xaxes(
                range=[model_prediction.variable_array.min(), model_prediction.variable_array.max()],
                row=row_no,
                col=col_no
            )

            if col_no == 1:
                figure.update_yaxes(title_text=objective_names[objective_index], row=row_no, col=col_no)

            if row_no == n_objectives:
                figure.update_xaxes(title_text=variable_names[variable_index], row=row_no, col=col_no)

    figure.update_layout(
        title={'text': f"Points and predictions {model_prediction.title}"}
    )

    return figure


def run_prediction_grid_app(
        optimiser: BayesianOptimiser
) -> None:

    @callback(
        Output('prediction-grid', 'figure'),
        Input('button-go-to-point', 'n_clicks'),
        State('dropdown-points', 'value'),
    )
    def update_x_timeseries(
            n_clicks: int,
            point_index: int
    ) -> go.Figure:

        if point_index is None:
            raise PreventUpdate()

        else:

            chosen_point = variable_values[point_index]

            figure = plot_prediction_grid_from_optimiser(
                optimiser=optimiser,
                return_figure=True,
                model_prediction_container=model_prediction_container,
                evaluated_point=chosen_point
            )

            assert figure is not None

        return figure

    n_points_evaluated = optimiser.n_points_evaluated

    if optimiser.suggested_points is None:

        variable_values = optimiser.evaluated_variable_values.tensor
        point_names = [f"Point. {point_no}" for point_no in range(0, n_points_evaluated)]

    else:

        n_suggested_points: int = optimiser.n_evaluations_per_step

        variable_values = torch.concat([
            optimiser.evaluated_variable_values.tensor,
            optimiser.suggested_points.variable_values
        ])

        suggested_point_names = [f"Suggested point no. {point_no}" for point_no in range(n_suggested_points)]

        point_names = (
            [f"Point no. {point_no}" for point_no in range(n_points_evaluated)] + suggested_point_names
        )

    model_prediction_container = ModelPredictionContainer()

    fig_1 = plot_prediction_grid_from_optimiser(
        optimiser=optimiser,
        return_figure=True,
        model_prediction_container=model_prediction_container
    )

    dropdown_options = [{'label': point_names[i], 'value': i} for i in range(len(point_names))]

    app = Dash()

    app.layout = html.Div([  # type: ignore[misc]
        html.Div([
            dcc.Graph(
                id='prediction-grid',
                figure=fig_1,
                style={'height': '800px'}
            )
        ]),
        html.Div([
            html.Button(
                'Go to point',
                id='button-go-to-point',
                n_clicks=0
            ),
            dcc.Dropdown(
                id='dropdown-points',
                options=dropdown_options  # type: ignore[arg-type]
            )
        ])
    ])

    app.run()
