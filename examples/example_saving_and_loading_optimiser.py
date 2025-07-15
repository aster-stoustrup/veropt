from veropt import bayesian_optimiser, load_optimiser_from_state, save_to_json
from veropt.optimiser.practice_objectives import Hartmann

objective = Hartmann(
    n_variables=6
)

optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective
)

optimiser.run_optimisation_step()
optimiser.run_optimisation_step()

save_to_json(
    optimiser,
    file_name='test'
)

reloaded_optimiser = load_optimiser_from_state(
    'test.json'
)
