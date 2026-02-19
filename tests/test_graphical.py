import csv
import tempfile
from pathlib import Path

from veropt.optimiser.constructors import bayesian_optimiser
from veropt.optimiser.practice_objectives import VehicleSafety
from veropt.graphical.visualisation import save_table_to_csv


def test_save_table_to_csv() -> None:

    objective = VehicleSafety()

    optimiser = bayesian_optimiser(
        n_initial_points=32,
        n_bayesian_points=16,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
    )

    for _ in range(4):
        optimiser.run_optimisation_step()

    chosen_points = [0, 1, 10, 12]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "table.csv"

        save_table_to_csv(
            optimiser=optimiser,
            chosen_points=chosen_points,
            file_path=csv_path
        )

        with open(csv_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

    variable_names = optimiser.objective.variable_names
    objective_names = optimiser.objective.objective_names
    evaluated_variables = optimiser.evaluated_variables_real_units
    evaluated_objectives = optimiser.evaluated_objectives_real_units

    # Split rows into variable rows, separator, and objective rows
    variable_rows = rows[:len(variable_names)]
    separator_row = rows[len(variable_names)]
    objective_rows = rows[len(variable_names) + 1:]

    assert separator_row["Name"] == "Objectives"

    # Check variable rows
    for var_idx, var_name in enumerate(variable_names):
        assert variable_rows[var_idx]["Name"] == var_name
        for point_idx in chosen_points:
            expected_value = float(evaluated_variables[point_idx, var_idx])
            csv_value = float(variable_rows[var_idx][f"point_{point_idx}"])
            assert abs(csv_value - expected_value) < 1e-4, (
                f"Variable '{var_name}' at point {point_idx}: "
                f"expected {expected_value}, got {csv_value}"
            )

    # Check objective rows
    for obj_idx, obj_name in enumerate(objective_names):
        assert objective_rows[obj_idx]["Name"] == obj_name
        for point_idx in chosen_points:
            expected_value = float(evaluated_objectives[point_idx, obj_idx])
            csv_value = float(objective_rows[obj_idx][f"point_{point_idx}"])
            assert abs(csv_value - expected_value) < 1e-4, (
                f"Objective '{obj_name}' at point {point_idx}: "
                f"expected {expected_value}, got {csv_value}"
            )
