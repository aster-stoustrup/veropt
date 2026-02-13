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
reference_point_variables = {'var_1': 0.5, 'var_2': 0.5, 'var_3': 0.5}
variable_order = [reference_point_variables[name] for name in objective.variable_names]
reference_point_objectives_tensor = objective(torch.tensor([variable_order]))
reference_point_objectives = {
    name: float(value) for name, value in
    zip(objective.objective_names, reference_point_objectives_tensor[0])
}

reference_point_data = {
    'variable_values': reference_point_variables,
    'objective_values': reference_point_objectives
}

optimiser.add_reference_point_real_units(reference_point_data)

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
