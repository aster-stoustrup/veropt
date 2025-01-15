from sklearn.manifold import TSNE
import torch
import plotly.graph_objs as go


# TODO: Probably separate the plotting aspect from calculating the projection
#   - More importantly, t-SNE might not do that much for us here
def map_coordinates_into_tsne(
        obj_func_coords: torch.tensor,
        obj_func_vals: torch.tensor
):
    combined_coords_vals = torch.concat([obj_func_coords, obj_func_vals], dim=1)

    tsne = TSNE(n_components=2)
    tsne_projection = tsne.fit_transform(combined_coords_vals)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tsne_projection[:, 0],
        y=tsne_projection[:, 1],
        mode='markers'
    ))

    fig.show()


