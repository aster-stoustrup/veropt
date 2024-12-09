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


def display_suggested_points_from_optimiser(
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

    display_suggested_points(
        obj_func_coords=obj_func_coords,
        obj_func_vals=obj_func_vals,
        obj_names=obj_names,
        var_names=var_names,
        shown_inds=shown_inds
    )



# TODO: Untangle all visualisation tools from god object and put them in here


# TODO: Add type hints
# TODO: Find better name, could also be used for evaluated points
#   - When we do this, also need to rename input names
def display_suggested_points(
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


def plot_correlations(
        obj_func_coords,
        obj_func_vals,
        obj_names,
        var_names_in=None
):
    # Note: This function will generally probably want all evaluated points (?)

    # Note: If there's only one or two objectives but many pars, maybe do two subfigs to avoid tall and narrows figs

    # obj_func_coords = optimiser.obj_func_coords.squeeze(0)
    # obj_func_vals = optimiser.obj_func_vals.squeeze(0)
    #
    # obj_names = optimiser.obj_func.obj_names

    n_suggested_steps = obj_func_coords.shape[0]
    n_params = obj_func_coords.shape[1]
    n_objs = len(obj_names)

    if var_names_in is None:
        var_names = [f"Par. {param_no}" for param_no in range(1, n_params + 1)]
    else:
        var_names = var_names_in

    # TODO: Do the same color scale for both graphs...?
    #   - But note that we're probably doing all points for this one and only some for the other one

    # TODO: Separate bayes point from init points

    # TODO: Add suggested point at red dots with uncertainty bars?
    #   - Is there any way we could show acq func values anywhere...?!
    #       - Prooobably not since those depend on all parameters...? Might be confusing...?

    # TODO: If we're extremely cool and rad, we could make a dash app...
    #   - With sliders for each parameters and model predictions
    #       - Would need to have a button to not calculate every percent change probably (even though that would be rad)
    #   - And then one could slide through the dependencies of pars and objs
    #   - Could also have a thing where you can go to a particular evaluated point and see dependencies in that point
    #   - And we can do the dist fade thing to always show how far we are from different points

    fig = make_subplots(
        rows=n_objs,
        cols=n_params
    )

    for obj_no in range(n_objs):
       for param_no in range(n_params):

            row_no = n_objs - obj_no  # Placing these backwards to make the "y axes" of subplots go positive upwards
            col_no = param_no + 1

            # TODO: Add hover to show point no
            fig.add_trace(
            go.Scatter(
                x=obj_func_coords[:, param_no],
                y=obj_func_vals[:, obj_no],
                mode='markers',
                # TODO: Check that the colors are right
                #   - Might be easier if we implement the hover...?
                #   - As long as the hover works probably B)
                marker={'color': list(range(n_suggested_steps))},
                showlegend=False
            ),
            row=row_no,
            col=col_no
            )

            if col_no == 1:
                fig.update_yaxes(title_text=obj_names[obj_no], row=row_no, col=col_no)

            if row_no == n_objs:
                fig.update_xaxes(title_text=var_names[param_no], row=row_no, col=col_no)

    # TODO: Add legend
    #   - Maybe go see what they did in the SPLOMS
    #       - What an insane name for a plot type :))

    fig.show()


# TODO: Find a nice home for this sweet little guy
class ModelPrediction:
    def __init__(self, calc_pred_output: tuple, var_ind: int):

        self.var_ind = var_ind

        # TODO: Should do this nicer
        #   - If calc_pred should actually output all this, let's make it a dict?
        self.title: str = calc_pred_output[0]
        self.var_arr: np.ndarray =  calc_pred_output[1]
        self.model_mean_list: list[torch.tensor] = calc_pred_output[2]
        self.model_lower_std_list: list[torch.tensor] = calc_pred_output[3]
        self.model_upper_std_list: list[torch.tensor] = calc_pred_output[4]
        self.acq_fun_vals: np.ndarray = calc_pred_output[5]
        self.fun_arr: torch.tensor = calc_pred_output[6]
        self.samples: torch.tensor = calc_pred_output[8]

        self.point: torch.tensor = calc_pred_output[7]


# TODO: Find a nice home for this sweet little guy
class ModelPredictionContainer:
    def __init__(self):
        self.data: list[ModelPrediction] = []
        self.points: torch.tensor = torch.tensor([])
        self.var_inds: np.ndarray = np.array([])

    def add_data(self, model_prediction: ModelPrediction):
        self.data.append(model_prediction)

        if self.points.numel() == 0:
            self.points = model_prediction.point.unsqueeze(0)
        else:
            self.points = torch.concat([self.points, model_prediction.point.unsqueeze(0)], dim=0)

        self.var_inds = np.append(self.var_inds, model_prediction.var_ind)

    def __getitem__(self, data_ind: int) -> ModelPrediction:
        return self.data[data_ind]

    def locate_data(self, var_ind: int, point: torch.tensor) -> int | None:

        # Can we do without the mix of np and torch here?
        matching_var_inds = torch.tensor(np.equal(var_ind, self.var_inds))

        # NB: Not using any tolerance at the moment, might make this a little unreliable
        no_matching_coordinates_per_point = torch.eq(point, self.points).sum(dim=1)

        n_vars = self.points.shape[1]

        matching_points = no_matching_coordinates_per_point == n_vars

        matching_point_and_var = matching_var_inds * matching_points

        full_match_ind = torch.where(matching_point_and_var)[0]

        if len(full_match_ind) == 1:
            return int(full_match_ind)

        elif full_match_ind.numel() == 0:
            return None

        elif len(full_match_ind) > 1:
            raise RuntimeError("Found more than one matching point.")

        else:
            raise RuntimeError("Unexpected error.")

    def __call__(self, var_ind, point) -> ModelPrediction:
        data_ind = self.locate_data(
            var_ind=var_ind,
            point=point
        )

        if data_ind is None:
            raise ValueError("Point not found.")

        return self.data[data_ind]

    def __contains__(self, point):

        # Just checking if it has it for var_ind = 0, might be sensible to make it a bit more general/stable
        data_ind = self.locate_data(
            var_ind=0,
            point=point
        )

        if data_ind is None:
            return False

        elif type(data_ind) == int:
            return True

        else:
            raise RuntimeError


# TODO: Figure out naming and location
def prediction_grid_from_optimiser(
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

    if return_fig:

        fig = plot_prediction_grid(
            model_prediction_container=model_prediction_container,
            evaluated_point=evaluated_point,
            obj_func_coords=obj_func_coords,
            obj_func_vals=obj_func_vals,
            obj_names=obj_names,
            var_names=var_names,
            return_fig=True
        )

        return fig

    else:

        plot_prediction_grid(
            model_prediction_container=model_prediction_container,
            evaluated_point=evaluated_point,
            obj_func_coords=obj_func_coords,
            obj_func_vals=obj_func_vals,
            obj_names=obj_names,
            var_names=var_names
        )


def plot_prediction_grid(
        model_prediction_container: ModelPredictionContainer,
        evaluated_point: torch.Tensor,
        obj_func_coords: torch.Tensor,
        obj_func_vals: torch.Tensor,
        obj_names: list[str],
        var_names: list[str],
        return_fig: bool = False
):
    # TODO: Add option to plot subset of all these
    #   - Could be from var/obj start_ind to var/obj end_ind
    #   - Could be lists of vars and objs
    #   - Could be single var or single obj
    #   - Could be mix of these

    n_points = obj_func_coords.shape[0]
    n_params = obj_func_coords.shape[1]
    n_objs = len(obj_names)

    color_scale = colors.get_colorscale('Inferno')
    color_list = colors.sample_colorscale(
        colorscale=color_scale,
        samplepoints=n_points,
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

        opacity_list, distance_list = opacity_for_multidimensional_points(
            var_ind=var_ind,
            n_params=n_params,
            coordinates=obj_func_coords.unsqueeze(0),
            evaluated_point=evaluated_point,
            alpha_min=0.3,
            alpha_max=0.9
        )

        color_list_w_opacity = [
            "rgba(" + color_list[point_no][4:-1] + f", {opacity_list[point_no]})" for point_no in range(n_points)
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
                customdata=np.dstack([list(range(n_points)), distance_list])[0],
                hovertemplate="Param. value: %{x:.3f} <br>"
                              "Obj. func. value: %{y:.3f} <br>"
                              "Point number: %{customdata[0]:.0f} <br>"
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

    if return_fig:
        return fig
    else:
        fig.show()

    # TODO: Add version with suggested points


def prediction_grid_app(
        optimiser: BayesOptimiser
):

    @callback(
        Output('prediction-grid', 'figure'),
        Input('button-go-to-point', 'n_clicks'),
        State('dropdown-points', 'value'),
    )
    def update_x_timeseries(n_clicks: int, chosen_point_string: str):

        if chosen_point_string is None:
            raise PreventUpdate

        else:

            # NB: finding the integer in this string is hard-coded atm and will break if string is changed
            point_index = int(chosen_point_string[7:])

            chosen_point = obj_func_coords[point_index]

            fig = prediction_grid_from_optimiser(
                optimiser=optimiser,
                return_fig=True,
                model_prediction_container=model_prediction_container,
                evaluated_point=chosen_point
            )

        return fig

    obj_func_coords = optimiser.obj_func_coords.squeeze(0)

    n_points_evaluated = optimiser.n_points_evaluated
    point_names = [f"Point. {point_no}" for point_no in range(0, n_points_evaluated)]

    model_prediction_container = ModelPredictionContainer()

    fig_1 = prediction_grid_from_optimiser(
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
                options=point_names
            )
        ])
    ])

    app.run()

