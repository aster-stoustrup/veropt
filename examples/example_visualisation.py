from veropt import bayesian_optimiser
from veropt.graphical.visualisation import (
    plot_progression_from_optimiser,
    plot_pareto_front_grid_from_optimiser,
    plot_prediction_grid_from_optimiser,
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

    # TODO: Add more?

    progression_figure.show()
    pareto_front_grid.show()
    prediction_figure.show()

    if save:

        # TODO: Save figures

        pass


objective = VehicleSafety()

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective
)

for i in range(4):
    optimiser.run_optimisation_step()

optimiser.suggest_candidates()

make_figures(
    save=False
)


# TODO: Finish this example
#   - Maybe prevent acq func optimiser from taking too long
