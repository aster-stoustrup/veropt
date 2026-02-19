import torch

from veropt import bayesian_optimiser
from veropt.graphical.visualisation import (
    plot_pareto_front, plot_point_overview, plot_prediction_grid,
    plot_progression, plot_table
)
from veropt.optimiser.practice_objectives import DTLZ1

objective = DTLZ1(n_variables=3, n_objectives=2)

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=8,
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

# Evaluate the objective at the reference point
reference_point_variable_values = torch.tensor([[0.5, 0.1, 0.8]])
reference_point_objective_values = optimiser.objective(reference_point_variable_values)

reference_point_data = {
    'variable_values': {
        'var_1': reference_point_variable_values[0, 0],
        'var_2': reference_point_variable_values[0, 1],
        'var_3': reference_point_variable_values[0, 2]
    },
    'objective_values': {
        'DTLZ1 1': reference_point_objective_values[0, 0],
        'DTLZ1 2': reference_point_objective_values[0, 1]
    }
}

optimiser.add_reference_point_real_units(reference_point_data)  # type: ignore[arg-type]

for iteration in range(5):
    optimiser.run_optimisation_step()

# Print parameter table
chosen_point_indices = [0, 5, 10]
# Note: You can also just build the table without plotting it, e.g. to use it for a paper
# table = build_table(
#     optimiser=optimiser,
#     chosen_points=chosen_point_indices
# )
table_figure = plot_table(
    optimiser=optimiser,
    chosen_points=chosen_point_indices
)
table_figure.show()


# TODO: Also do a pareto grid...?
#   - Maybe just once to look at it before releasing this branch
figure_pareto = plot_pareto_front(
    optimiser=optimiser,
    plotted_objective_indices=[0, 1]
)
figure_pareto.show()

figure_prediction = plot_prediction_grid(optimiser=optimiser)
figure_prediction.show()

figure_overview = plot_point_overview(optimiser=optimiser)
figure_overview.show()

figure_progression = plot_progression(optimiser=optimiser)
figure_progression.show()
