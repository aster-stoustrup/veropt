from veropt import bayesian_optimiser
from veropt.graphical.visualisation import (
    plot_progression,
    plot_pareto_front_grid,
    plot_prediction_grid,
    plot_prediction_surface_grid
)
from veropt.optimiser.practice_objectives import VehicleSafety


def make_figures(
        show: bool,
        save_html: bool,
        save_pdf: bool
):

    prediction_surfaces = {}
    for objective_name in optimiser.objective.objective_names:
        prediction_surfaces[objective_name] = plot_prediction_surface_grid(
            optimiser=optimiser,
            objective=objective_name,
            return_figure=True,
            # n_points_per_dimension=50  # This can be set to decrease computational demand (but lower resoution)
        )

    if show:

        for objective_name in optimiser.objective.objective_names:
            prediction_surfaces[objective_name].show()

    if save_html:

        for objective_name in optimiser.objective.objective_names:
            prediction_surfaces[objective_name].write_html(f'prediction_surface{objective_name}.html')

    if save_pdf:

        # Note that at the time of making this example (Dec, 2025), there seems to be a bug in creating this as an image
        #   - Hopefully plotly/kaleido will rectify this soon

        for objective_name in optimiser.objective.objective_names:
            prediction_surfaces[objective_name].write_image(f'prediction_surface{objective_name}.pdf')


objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=120,
    n_bayesian_points=240,
    n_evaluations_per_step=4,
    objective=objective,
    model={
        'training_settings': {  # Setting this restriction so the model trains faster (not recommended)
            'max_iter': 50
        }
    },
    acquisition_optimiser={
        'optimiser': 'dual_annealing',
        'optimiser_settings': {  # Setting this restriction so new points are found quickly (not recommended)
            'max_iter': 50
        }
    }
)

for i in range(31):
    optimiser.run_optimisation_step()

optimiser.suggest_candidates()

# Please note that making these 3d figures can be computationally expensive

make_figures(
    show=True,
    save_html=True,
    save_pdf=False
)
