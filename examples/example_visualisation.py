from veropt import bayesian_optimiser
from veropt.graphical.visualisation import (
    plot_progression_from_optimiser,
    plot_pareto_front_grid_from_optimiser,
    plot_prediction_grid_from_optimiser,
    plot_prediction_surface_grid_from_optimiser
)
from veropt.optimiser.practice_objectives import VehicleSafety


def make_figures(
        save: bool
):

    progression_figure = plot_progression_from_optimiser(
        optimiser=optimiser,
        return_figure=True
    )

    pareto_front_grid = plot_pareto_front_grid_from_optimiser(
        optimiser=optimiser,
        return_figure=True
    )

    prediction_figure = plot_prediction_grid_from_optimiser(
        optimiser=optimiser,
        return_figure=True
    )

    prediction_surfaces = {}
    for objective_name in optimiser.objective.objective_names:
        prediction_surfaces[objective_name] = plot_prediction_surface_grid_from_optimiser(
            optimiser=optimiser,
            objective=objective_name,
            return_figure=True
        )

    progression_figure.show()
    pareto_front_grid.show()
    prediction_figure.show()

    for objective_name in optimiser.objective.objective_names:
        prediction_surfaces[objective_name].show()

    if save:

        # TODO: Save figures

        pass


objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
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

for i in range(4):
    optimiser.run_optimisation_step()

optimiser.suggest_candidates()

make_figures(
    save=False
)


# TODO: Finish this example
#   - Maybe prevent acq func optimiser from taking too long
