from veropt.optimiser.optimiser_saver_loader import load_optimiser_from_settings
from veropt.optimiser.practice_objectives import Hartmann


objective = Hartmann(
    n_variables=6
)


optimiser = load_optimiser_from_settings(
    file_name='example_1_optimiser',
    objective=objective,
)
