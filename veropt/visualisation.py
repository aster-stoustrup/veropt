from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
import plotly.graph_objs as go
import torch
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from plotly import colors
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from veropt import BayesOptimiser
from veropt.utility import opacity_for_multidimensional_points, get_best_points
from veropt.containers import SuggestedPoint, ModelPrediction, ModelPredictionContainer


# TODO: Decide on location
def plot_point_overview_from_optimiser(
        optimiser: BayesOptimiser,
        points: Literal['all', 'bayes', 'suggested', 'best']
):
    n_objs = optimiser.n_objs
    n_params = optimiser.n_params

    shown_inds = None

    if points == 'all':
        obj_func_coords = optimiser.obj_func_coords.squeeze(0)
        obj_func_vals = optimiser.obj_func_vals.squeeze(0)

    elif points == 'bayes':

        obj_func_coords = optimiser.obj_func_coords.squeeze(0)
        obj_func_vals = optimiser.obj_func_vals.squeeze(0)

        shown_inds = np.arange(optimiser.n_init_points, optimiser.n_points_evaluated)

    elif points == 'suggested':
        obj_func_coords = optimiser.suggested_steps.squeeze(0)

        n_points = obj_func_coords.shape[0]

        # TODO: Add some in_real_units support somewhere probably
        expected_values = [0.0] * n_points

        # TODO: Use the calc_suggested method instead
        # TODO: Add variance?
        for point_no in range(n_points):
        # Consider if evaluating the model should be a method in the optimiser class
            expected_value_list = optimiser.model.eval(optimiser.suggested_steps[:, point_no])
            expected_values[point_no] = torch.cat([val.loc for val in expected_value_list], dim=1).squeeze(0).detach().numpy()

        obj_func_vals = torch.tensor(np.array(expected_values))

    elif points == 'best':

        # TODO: Might be optimal to open all points but mark the best ones or make them visible or something

        best_inds = []

        best_inds.append(get_best_points(optimiser=optimiser)[2])

        for obj_ind in range(n_objs):

            best_inds.append(
                get_best_points(
                    optimiser=optimiser,
                    best_for_obj_ind=obj_ind
                )[2]
            )

        shown_inds = np.unique(best_inds)

        obj_func_coords = optimiser.obj_func_coords.squeeze(0)
        obj_func_vals = optimiser.obj_func_vals.squeeze(0)

    else:
        raise ValueError

    obj_names = optimiser.obj_func.obj_names

    if optimiser.obj_func.var_names is None:
        var_names = [f"Parameter {param_no}" for param_no in range(1, n_params + 1)]
    else:
        var_names = optimiser.obj_func.var_names

    plot_point_overview(
        obj_func_coords=obj_func_coords,
        obj_func_vals=obj_func_vals,
        obj_names=obj_names,
        var_names=var_names,
        shown_inds=shown_inds,

    )


# TODO: Untangle all visualisation tools from god object and put them in here


# TODO: Add type hints
# TODO: Find better name, could also be used for evaluated points
#   - When we do this, also need to rename input names
def plot_point_overview(
        obj_func_coords,
        obj_func_vals,
        obj_names,
        var_names,
        shown_inds = None
):
    # TODO: Maybe want a longer colour scale to avoid duplicate colours...?
    color_scale = colors.qualitative.T10
    color_scale = colors.convert_colors_to_same_type(color_scale, colortype="rgb")[0]
    n_colors = len(color_scale)

    # TODO: Cool hover shit?
    #   - Even without a dash app, we could add the "sum score" for each point on hover

    n_points = obj_func_coords.shape[0]

    opacity_lines = 0.2

    fig = make_subplots(rows=2, cols=1)

    # TODO: Give the point numbers of all evaluated points (unless it's suggested points?)
    for point_no in range(n_points):

        if shown_inds is not None:
            if not point_no in shown_inds:
                args = {'visible': 'legendonly'}
            else:
                args = {}
        else:
            args = {}

        fig.add_trace(
            go.Scatter(
                x=var_names,
                y=obj_func_coords[point_no],
                name=f"Point no. {point_no}",  # This is currently out of the ones plotted, consider that
                line={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", {opacity_lines})"},
                marker={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", 1.0)"},
                mode='lines+markers',
                legendgroup=point_no,
                **args
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=obj_names,
                y=obj_func_vals[point_no],
                line={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", {opacity_lines})"},
                marker={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", 1.0)"},
                name=f"Point no. {point_no}",
                mode='lines+markers',
                legendgroup=point_no,
                showlegend=False,
                **args
            ),
            row=2,
            col=1
        )

    fig.update_layout(
        # title={'text': "Plot Title"},
        # xaxis={'title': {'text': "Parameter Number"}},  # Maybe obvious and unnecessary?
        yaxis={'title': {'text': "Parameter Values"}},  # TODO: Add if they're normalised or not
        # TODO: Add if they're predicted or evaluated
        yaxis2={'title': {'text': "Objective Values"}},  # TODO: Add if they're normalised or not
    )

    if n_points < 7:
        fig.update_layout(hovermode="x")

    fig.show()


# TODO: Decide on location
def plot_prediction_grid_from_optimiser(
        optimiser: BayesOptimiser,
        return_fig: bool = False,
        model_prediction_container: ModelPredictionContainer = None,
        evaluated_point: torch.tensor = None
):
    obj_func_coords = optimiser.obj_func_coords.squeeze(0)
    obj_func_vals = optimiser.obj_func_vals.squeeze(0)
    obj_names = optimiser.obj_func.obj_names

    n_params = obj_func_coords.shape[1]

    if optimiser.obj_func.var_names is None:
        var_names = [f"Par. {param_no}" for param_no in range(1, n_params + 1)]
    else:
        var_names = optimiser.obj_func.var_names

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
        for var_ind in range(n_params):

            calc_pred_out_tuple = optimiser.calculate_prediction(
                var_ind=var_ind,
                eval_point=evaluated_point
            )
            calculated_prediction = ModelPrediction(
                calc_pred_output=calc_pred_out_tuple,
                var_ind=var_ind
            )
            model_prediction_container.add_data(
                model_prediction=calculated_prediction
            )

    if evaluated_point is None:
        evaluated_point = calculated_prediction.point

    if optimiser.need_new_suggestions is False:
        suggested_point_predictions = optimiser.calculate_prediction_suggested_steps()
    else:
        suggested_point_predictions = None

    fig = plot_prediction_grid(
        model_prediction_container=model_prediction_container,
        evaluated_point=evaluated_point,
        obj_func_coords=obj_func_coords,
        obj_func_vals=obj_func_vals,
        obj_names=obj_names,
        var_names=var_names,
        suggested_points=suggested_point_predictions
    )

    if return_fig:

        return fig

    else:

        fig.show()


def plot_prediction_grid(
        model_prediction_container: ModelPredictionContainer,
        evaluated_point: torch.Tensor,
        obj_func_coords: torch.Tensor,
        obj_func_vals: torch.Tensor,
        obj_names: list[str],
        var_names: list[str],
        suggested_points: list[SuggestedPoint] = None
) -> go.Figure:
    # TODO: Add option to plot subset of all these
    #   - Could be from var/obj start_ind to var/obj end_ind
    #   - Could be lists of vars and objs
    #   - Could be single var or single obj
    #   - Could be mix of these

    n_evaluated_points = obj_func_coords.shape[0]
    if suggested_points:
        n_suggested_points = len(suggested_points)
    n_params = obj_func_coords.shape[1]
    n_objs = len(obj_names)

    color_scale = colors.get_colorscale('Inferno')
    color_list = colors.sample_colorscale(
        colorscale=color_scale,
        samplepoints=n_evaluated_points,
        low=0.0,
        high=1.0,
        colortype='rgb'
    )

    fig = make_subplots(
        rows=n_objs,
        cols=n_params
    )

    for var_ind in range(n_params):

        model_pred_data = model_prediction_container(
            var_ind=var_ind,
            point=evaluated_point
        )

        if suggested_points:
            joint_points = torch.concat([
                obj_func_coords,
                *[suggested_point.coordinates.unsqueeze(0) for suggested_point in suggested_points]
            ])
        else:
            joint_points = obj_func_coords

        joint_opacity_list, joint_distance_list = opacity_for_multidimensional_points(
            var_ind=var_ind,
            n_params=n_params,
            coordinates=joint_points.unsqueeze(0),
            evaluated_point=evaluated_point,
            alpha_min=0.3,
            alpha_max=0.9
        )

        distance_list = joint_distance_list[:n_evaluated_points]
        suggested_point_distance_list = joint_distance_list[n_evaluated_points:]

        color_list_w_opacity = [
            "rgba(" + color_list[point_no][4:-1] + f", {joint_opacity_list[point_no]})"
            for point_no in range(n_evaluated_points)
        ]

        if suggested_points:
            suggested_point_color_list_wo = [
                f"rgba(139, 0, 0, {joint_opacity_list[n_evaluated_points + point_no]})"
                for point_no in range(n_suggested_points)
            ]

        for obj_ind in range(n_objs):
            model_mean = model_pred_data.model_mean_list[obj_ind]
            model_lower_std = model_pred_data.model_lower_std_list[obj_ind]
            model_upper_std = model_pred_data.model_upper_std_list[obj_ind]

            row_no = n_objs - obj_ind  # Placing these backwards to make the "y axes" of subplots go positive upwards
            col_no = var_ind + 1

            fig.add_trace(
            go.Scatter(
                x=model_pred_data.var_arr,
                y=model_upper_std,
                line={'width': 0.0, 'color': 'rgba(156, 156, 156, 0.4)'},
                name='Upper bound prediction',
                showlegend=False
            ),
            row=row_no, col=col_no
            )

            fig.add_trace(
            go.Scatter(
                x=model_pred_data.var_arr,
                y=model_lower_std,
                fill='tonexty',  # This fills between this and the line above
                line={'width': 0.0, 'color': 'rgba(156, 156, 156, 0.4)'},
                name='Lower bound prediction',
                showlegend=False,
            ),
            row=row_no, col=col_no
            )

            fig.add_trace(
            go.Scatter(
                x=model_pred_data.var_arr,
                y=model_mean,
                line={'color': 'black'},
                name='Mean prediction',
                showlegend=False
            ),
            row=row_no, col=col_no
            )

            fig.add_trace(
            go.Scatter(
                x=obj_func_coords[:, var_ind],
                y=obj_func_vals[:, obj_ind],
                mode='markers',
                marker={'color': color_list_w_opacity},
                name='Evaluated point',
                showlegend=False,
                customdata=np.dstack([list(range(n_evaluated_points)), distance_list])[0],
                hovertemplate="Param. value: %{x:.3f} <br>"
                              "Obj. func. value: %{y:.3f} <br>"
                              "Point number: %{customdata[0]:.0f} <br>"
                              "Distance to current point: %{customdata[1]:.3f}"
            ),
            row=row_no, col=col_no
            )

            if suggested_points:
                for suggested_point_no, point in enumerate(suggested_points):
                    fig.add_trace(
                        go.Scatter(
                            x=point.coordinates[var_ind].detach().numpy(),
                            y=point.predicted_values[obj_ind].detach().numpy(),
                            error_y={
                                'type': 'data',
                                'symmetric': False,
                                'array': point.predicted_values_upper[obj_ind].detach().numpy(),
                                'arrayminus': point.predicted_values_lower[obj_ind].detach().numpy(),
                                'color': suggested_point_color_list_wo[suggested_point_no]
                            },
                            mode='markers',
                            marker={'color': suggested_point_color_list_wo[suggested_point_no]},
                            name='Suggested point',
                            showlegend=False,
                            customdata=np.dstack([
                                [suggested_point_no],
                                [suggested_point_distance_list[suggested_point_no]],
                                [point.predicted_values_upper[obj_ind]],
                                [point.predicted_values_lower[obj_ind]]
                            ])[0],
                            # TODO: Super sweet feature would be to check if upper and lower are equal and then do pm
                            hovertemplate="Param. value: %{x:.3f}"
                                          " + %{customdata[2]:.3f} /"
                                          " - %{customdata[3]:.3f}"
                                          "<br>"
                                          "Obj. func. value: %{y:.3f} <br>"
                                          "Suggested point number: %{customdata[0]:.0f} <br>"
                                          "Distance to current point: %{customdata[1]:.3f}"
                        ),
                        row=row_no, col=col_no
                    )

            fig.update_xaxes(
                range=[model_pred_data.var_arr.min(), model_pred_data.var_arr.max()],
                row=row_no,
                col=col_no
            )

            if col_no == 1:
                fig.update_yaxes(title_text=obj_names[obj_ind], row=row_no, col=col_no)

            if row_no == n_objs:
                fig.update_xaxes(title_text=var_names[var_ind], row=row_no, col=col_no)

    fig.update_layout(
        title={'text': f"Points and predictions {model_pred_data.title}"}
    )

    return fig


def prediction_grid_app(
        optimiser: BayesOptimiser
):

    @callback(
        Output('prediction-grid', 'figure'),
        Input('button-go-to-point', 'n_clicks'),
        State('dropdown-points', 'value'),
    )
    def update_x_timeseries(n_clicks: int, point_index: int):

        if point_index is None:
            raise PreventUpdate

        else:

            chosen_point = obj_func_coords[point_index]

            fig = plot_prediction_grid_from_optimiser(
                optimiser=optimiser,
                return_fig=True,
                model_prediction_container=model_prediction_container,
                evaluated_point=chosen_point
            )

        return fig

    if optimiser.need_new_suggestions is False:
        suggested_steps = True
        n_suggested_points = optimiser.n_evals_per_step
    else:
        suggested_steps = False

    n_points_evaluated = optimiser.n_points_evaluated

    if suggested_steps:
        obj_func_coords = torch.concat([
            optimiser.obj_func_coords.squeeze(0),
            optimiser.suggested_steps.squeeze(0)
        ])
    else:
        obj_func_coords = optimiser.obj_func_coords.squeeze(0)

    if suggested_steps:
        point_names = ([f"Point no. {point_no}" for point_no in range(n_points_evaluated)] +
                       [f"Suggested point no. {point_no}" for point_no in range(n_suggested_points)])
    else:
        point_names = [f"Point. {point_no}" for point_no in range(0, n_points_evaluated)]

    model_prediction_container = ModelPredictionContainer()

    fig_1 = plot_prediction_grid_from_optimiser(
        optimiser=optimiser,
        return_fig=True,
        model_prediction_container=model_prediction_container
    )

    app = Dash()

    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                id='prediction-grid',
                figure=fig_1,
                style={'height': '800px'}
            )
    ],
        ),
        html.Div([
            html.Button(
                'Go to point',
                id='button-go-to-point',
                n_clicks=0
            ),
            dcc.Dropdown(
                id='dropdown-points',
                options=[{'label': point_names[i], 'value': i} for i in range(len(point_names))]
            )
        ])
    ])

    app.run()

