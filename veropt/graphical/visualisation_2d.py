import warnings
from typing import Union

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.express import colors

from veropt.graphical.visualisation_utility import opacity_for_multidimensional_points, get_continuous_colour
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.utility import TensorWithNormalisationFlag


def plot_prediction_surface_from_optimiser(
        optimiser: BayesianOptimiser,
        variable_x: Union[int, str],
        variable_y: Union[int, str],
        objective: Union[int, str],
        evaluated_point: torch.Tensor,
        normalised: bool = False,
        n_points_per_dimension: int = 200
) -> go.Figure:

    if normalised is False:
        bounds = optimiser.bounds_real_units

    else:
        bounds = optimiser.bounds.tensor

    # TODO: Allow evaluated_point to be int or None
    #   - Re-use from 2d version

    if isinstance(variable_x, str):
        variable_x = optimiser.objective.variable_names.index(variable_x)

    if isinstance(variable_y, str):
        variable_y = optimiser.objective.variable_names.index(variable_y)

    if isinstance(objective, str):
        objective = optimiser.objective.objective_names.index(objective)

    variable_tensor_x = torch.linspace(
        start=bounds[0, variable_x],
        end=bounds[1, variable_x],
        steps=n_points_per_dimension
    )

    variable_tensor_y = torch.linspace(
        start=bounds[0, variable_y],
        end=bounds[1, variable_y],
        steps=n_points_per_dimension
    )

    grid_x, grid_y = torch.meshgrid(
    variable_tensor_x,
        variable_tensor_y,
        indexing='xy'
    )

    all_variables_tensor = evaluated_point.repeat(
        n_points_per_dimension * n_points_per_dimension,
        1
    )

    for i in range(n_points_per_dimension):
        all_variables_tensor[
        i * n_points_per_dimension:i * n_points_per_dimension + n_points_per_dimension,
        variable_x
        ] = grid_x[i, :]

        all_variables_tensor[
        i * n_points_per_dimension:i * n_points_per_dimension + n_points_per_dimension,
        variable_y
        ] = grid_y[i, :]

    prediction = optimiser.predictor.predict_values(
        variable_values=all_variables_tensor,
        normalised=normalised
    )

    prediction_objective_tensor = prediction['mean'][:, objective]

    prediction_objective_matrix = prediction_objective_tensor.reshape(n_points_per_dimension, n_points_per_dimension)

    if normalised is False:
        point_variable_values = optimiser.evaluated_variables_real_units
        point_objective_values = optimiser.evaluated_objectives_real_units[:, objective]
    else:
        point_variable_values = optimiser.evaluated_variable_values.tensor
        point_objective_values = optimiser.evaluated_objective_values[:, objective].tensor

    x_axis_title = optimiser.objective.variable_names[variable_x]
    y_axis_title = optimiser.objective.variable_names[variable_y]
    z_axis_title = optimiser.objective.objective_names[objective]

    if normalised:
        x_axis_title += '(normalised)'
        y_axis_title += '(normalised)'
        z_axis_title += '(normalised)'

    return plot_prediction_surface(
        prediction_objective_matrix=prediction_objective_matrix,
        prediction_grid_x=grid_x,
        prediction_grid_y=grid_y,
        point_variable_values=point_variable_values,
        point_objective_values=point_objective_values,
        evaluated_point=evaluated_point,
        variable_x_index=variable_x,
        variable_y_index=variable_y,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        z_axis_title=z_axis_title
    )


def plot_prediction_surface(
        prediction_objective_matrix: torch.Tensor,
        prediction_grid_x: torch.Tensor,
        prediction_grid_y: torch.Tensor,
        point_variable_values: torch.Tensor,
        point_objective_values: torch.Tensor,
        evaluated_point: torch.Tensor,
        variable_x_index: int,
        variable_y_index: int,
        x_axis_title: str,
        y_axis_title: str,
        z_axis_title: str,
) -> go.Figure:

    # TODO: Add some colour/opacity things to show distance to plane

    colour_scale = colors.get_colorscale('matter')

    opacity_list, distance_list = opacity_for_multidimensional_points(
        variable_indices=[variable_x_index, variable_y_index],
        variable_values=point_variable_values,
        evaluated_point=evaluated_point,
        alpha_min=0.4,
        alpha_max=1.0
    )

    colour_list = [
        get_continuous_colour(
            colour_scale=colour_scale,
            value=float(1 - distance)
        ) for distance in distance_list
    ]

    colour_list_w_opacity = [
        "rgba(" + colour_list[point_no][4:-1] + f", {opacity_list[point_no]})"
        for point_no in range(len(opacity_list))
    ]

    figure = go.Figure()

    figure.add_trace(go.Surface(
        x=prediction_grid_x.detach().numpy(),
        y=prediction_grid_y.detach().numpy(),
        z=prediction_objective_matrix.detach().numpy(),
    ))

    figure.add_trace(go.Scatter3d(
        x=point_variable_values[:, variable_x_index].detach().numpy(),
        y=point_variable_values[:, variable_y_index].detach().numpy(),
        z=point_objective_values.detach().numpy(),
        mode='markers',
        marker={
            'color': colour_list_w_opacity,
            'opacity': 1.0
        },
    ))

    figure.update_layout(
        scene={
            'xaxis': {
                'title': {
                    'text': x_axis_title
                }
            },
            'yaxis': {
                'title': {
                    'text': y_axis_title
                }
            },
            'zaxis': {
                'title': {
                    'text': z_axis_title
                }
            }
        }
    )

    return figure
