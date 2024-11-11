import plotly.express as px
import plotly.graph_objs as go

# TODO: Untangle all visualisation tools from god object and put them in here


# TODO: Find better name, could also be used for evaluated points
def display_suggested_points(
        suggested_steps,
        predicted_obj_func_vals
):
    # TODO: Make method for this in optimiser?
    # optimiser.suggested_steps.squeeze(0).T

    fig = go.Figure()

    fig.add_trace(go.Scatter(

    ))

    # TODO: Probably change to go
    fig = px.scatter(
        suggested_steps,
        labels={'value': 'Variable value', 'index': 'Variable number', 'variable': 'suggested point no.'}
    )

    fig.show()

    # TODO: Add some in_real_units support somewhere probably
    # amount_of_points = optimiser.suggested_steps.shape[1]
    # expected_values = [0.0] * amount_of_points
    #
    # for point_no in range(amount_of_points):
    #     expected_value_list = optimiser.model.eval(optimiser.suggested_steps[:, point_no])
    #     expected_values[point_no] = torch.cat([val.loc for val in expected_value_list], dim=1).squeeze(0).detach().numpy()
    #
    # predicted_obj_func_vals = np.array(expected_values).T

    # Can probably add this once we go to go
    # obj_names = optimiser.obj_func.obj_names

    # TODO: Add variance
    fig = px.scatter(
        predicted_obj_func_vals,
        labels={'value': 'Predicted obj func value', 'index': 'Objective no.', 'variable': 'suggested point no.'}
    )

    fig.show()


