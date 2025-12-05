from veropt import bayesian_optimiser
from veropt.graphical.visualisation import (
    plot_progression,
    plot_pareto_front_grid,
    plot_prediction_grid,
    plot_point_overview
)
from veropt.optimiser.practice_objectives import VehicleSafety


def make_figures(
        show: bool,
        save_html: bool,
        save_pdf: bool
):

    progression_figure = plot_progression(
        optimiser=optimiser,
        return_figure=True
    )

    points_overview_figure = plot_point_overview(
        optimiser=optimiser,
        points='pareto-optimal',
        return_figure=True
    )

    pareto_front_grid = plot_pareto_front_grid(
        optimiser=optimiser,
        return_figure=True
    )

    prediction_figure = plot_prediction_grid(
        optimiser=optimiser,
        return_figure=True
    )

    if show:

        progression_figure.show()
        points_overview_figure.show()
        pareto_front_grid.show()
        prediction_figure.show()

    if save_html:

        progression_figure.write_html('progression.html')
        points_overview_figure.write_html('points_overview.html')
        pareto_front_grid.write_html('pareto_front_grid.html')
        prediction_figure.write_html('prediction_grid.html')

    if save_pdf:

        # Note that other image formats can also be used here
        #   - Simply change the file ending to save in a different format

        progression_figure.write_image('progression.pdf')
        points_overview_figure.write_image('points_overview.pdf')
        pareto_front_grid.write_image('pareto_front_grid.pdf')
        prediction_figure.write_image('prediction_grid.pdf')


objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=120,
    n_bayesian_points=240,
    n_evaluations_per_step=4,
    objective=objective,
    model={
        'training_settings': {  # Setting this restriction so the model trains worse but faster (not recommended)
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

make_figures(
    show=True,
    save_html=False,
    save_pdf=False
)
