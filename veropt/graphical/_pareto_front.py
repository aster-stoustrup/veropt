from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.express import colors
from plotly.subplots import make_subplots

from veropt.optimiser.optimiser_utility import SuggestedPoints
from veropt.optimiser.utility import DataShape


def _plot_pareto_front_grid(
        objective_values: torch.Tensor,
        objective_names: list[str],
        pareto_optimal_indices: list[int],
        n_initial_points: int,
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
                    objective_values=objective_values,
                    objective_index_x=objective_index_x,
                    objective_index_y=objective_index_y,
                    objective_names=objective_names,
                    pareto_optimal_indices=pareto_optimal_indices,
                    n_initial_points=n_initial_points,
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
        objective_names: list[str],
        pareto_optimal_indices: list[int],
        n_initial_points: int,
        suggested_points: Optional[SuggestedPoints] = None,
        row: Optional[int] = None,
        col: Optional[int] = None
) -> go.Figure:

    # Note: Must pass all points to this function or point numbers will be wrong

    n_evaluated_points = objective_values.shape[DataShape.index_points]
    point_numbers = np.arange(n_evaluated_points).reshape(n_evaluated_points, 1)

    pareto_point_numbers = point_numbers[pareto_optimal_indices]
    dominating_objective_values = objective_values[pareto_optimal_indices]

    if row is None and col is None:
        row_col_info = {}
        show_legend = True

    else:
        row_col_info = {
            'row': row,
            'col': col
        }

        if row == 1 and col == 1:
            show_legend = True
        else:
            show_legend = False

    color_scale = colors.qualitative.Plotly
    color_evaluated_points = color_scale[0]

    figure.add_trace(
        go.Scatter(
            x=objective_values[:n_initial_points, objective_index_x],
            y=objective_values[:n_initial_points, objective_index_y],
            mode='markers',
            name='Initial points',
            legendgroup='Initial points',
            showlegend=show_legend,
            marker={
                'symbol': 'diamond',
                'color': color_scale[2]
            },
            customdata=point_numbers[:n_initial_points],
            hovertemplate="Point number: %{customdata[0]:.0f} <br>"
                          f"{objective_names[objective_index_x]}: " + "%{x:.3f} <br>"
                          f"{objective_names[objective_index_y]}: " + "%{y:.3f} <br>"
        ),
        **row_col_info
    )

    figure.add_trace(
        go.Scatter(
            x=objective_values[n_initial_points:, objective_index_x],
            y=objective_values[n_initial_points:, objective_index_y],
            mode='markers',
            name='Bayesian points',
            legendgroup='Bayesian points',
            showlegend=show_legend,
            marker={'color': color_evaluated_points},
            customdata=point_numbers[n_initial_points:],
            hovertemplate="Point number: %{customdata[0]:.0f} <br>"
                          f"{objective_names[objective_index_x]}: " + "%{x:.3f} <br>"
                          f"{objective_names[objective_index_y]}: " + "%{y:.3f} <br>"
        ),
        **row_col_info
    )

    marker_type_dominating = np.array(['circle'] * len(pareto_optimal_indices)).astype('U20')
    dominating_initials = n_initial_points > np.array(pareto_optimal_indices)
    marker_type_dominating[dominating_initials] = 'diamond'
    marker_type_dominating_list = marker_type_dominating.tolist()

    # TODO: Change colour by mean of all objectives
    #   - Will have to normalise them first or it will be scale-dependant

    figure.add_trace(
        go.Scatter(
            x=dominating_objective_values[:, objective_index_x],
            y=dominating_objective_values[:, objective_index_y],
            mode='markers',
            marker={
                'color': 'black',
                'symbol': marker_type_dominating_list
            },
            name='Dominating points',
            legendgroup='Dominating points',
            showlegend=show_legend,
            customdata=pareto_point_numbers,
            hovertemplate="Point number: %{customdata[0]:.0f} <br>"
                          f"{objective_names[objective_index_x]}: " + "%{x:.3f} <br>"
                          f"{objective_names[objective_index_y]}: " + "%{y:.3f} <br>"
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
                    name='Suggested points',
                    legendgroup="Suggested points",
                    showlegend=True if (suggested_point_no == 0 and show_legend) else False,
                    customdata=np.dstack([
                                [prediction['lower'][objective_index_x].detach().numpy()],
                                [prediction['upper'][objective_index_x].detach().numpy()],
                                [prediction['lower'][objective_index_y].detach().numpy()],
                                [prediction['upper'][objective_index_y].detach().numpy()],
                    ])[0],
                    hovertemplate=f"Suggested point number: {suggested_point_no} <br>"
                                  f"{objective_names[objective_index_x]}: "
                                  "%{x:.3f} (%{customdata[0]:.3f} to %{customdata[1]:.3f}) <br>"
                                  f"{objective_names[objective_index_y]}: "
                                  "%{y:.3f} (%{customdata[2]:.3f} to %{customdata[3]:.3f}) <br>"
                ),
                **row_col_info
            )

    return figure


def _plot_pareto_front(
        objective_values: torch.Tensor,
        pareto_optimal_indices: list[int],
        plotted_objective_indices: list[int],
        objective_names: list[str],
        n_initial_points: int,
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
            objective_names=objective_names,
            pareto_optimal_indices=pareto_optimal_indices,
            n_initial_points=n_initial_points,
            suggested_points=suggested_points
        )

        figure.update_xaxes(
            title_text=objective_names[obj_ind_x]
        )

        figure.update_yaxes(
            title_text=objective_names[obj_ind_y]
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
