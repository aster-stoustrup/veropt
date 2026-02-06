from veropt import bayesian_optimiser
from veropt.optimiser.practice_objectives import DTLZ1
from veropt.graphical.visualisation import (
    plot_pareto_front, plot_pareto_front_grid,
    plot_prediction_grid,
    plot_point_overview,
    plot_progression
)


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

reference_point_data = {
    'variable_values': {
        'var_1': 0.5, 'var_2': 0.5, 'var_3': 0.5
    },
    'objective_values': {
        'DTLZ1 1': -50.0, 'DTLZ1 2': -50.0
    }
}
optimiser.add_reference_point_real_units(reference_point_data)

for iteration in range(5):
    optimiser.run_optimisation_step()

# TODO: Also do a pareto grid...?
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