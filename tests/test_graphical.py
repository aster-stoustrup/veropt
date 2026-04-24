import csv
import tempfile
from pathlib import Path

import torch

from veropt.optimiser.constructors import bayesian_optimiser
from veropt.optimiser.practice_objectives import VehicleSafety
from veropt.graphical.visualisation import save_table_to_csv, plot_pareto_front, plot_pareto_front_grid
from veropt.graphical._pareto_front import _add_pareto_traces_2d, _build_ellipse_traces


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


def _make_noisy_vehicle_safety_optimiser() -> object:
    """Helper: small VehicleSafety optimiser with noise_std set on all three objectives."""
    objective = VehicleSafety(
        noise_std={'VeSa 1': 0.1, 'VeSa 2': 0.1, 'VeSa 3': 0.05}
    )
    optimiser = bayesian_optimiser(
        n_initial_points=16,
        n_bayesian_points=8,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
        acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 50}},
        model={'training_settings': {'max_iter': 5}},
    )
    for _ in range(4):
        optimiser.run_optimisation_step()
    return optimiser


class TestUncertainParetoFront:

    def test_build_ellipse_traces_shape(self) -> None:
        """_build_ellipse_traces returns a single Scatter trace with None separators."""
        import numpy as np
        import plotly.graph_objects as go
        x_centres = np.array([0.0, 1.0, 2.0])
        y_centres = np.array([0.0, 1.0, 2.0])
        trace = _build_ellipse_traces(
            x_centres=x_centres,
            y_centres=y_centres,
            sigma_x=0.1,
            sigma_y=0.2,
            colour='rgba(100, 100, 100, 1.0)',
            name='test ellipses',
            show_legend=True,
            legend_group='test',
        )
        assert isinstance(trace, go.Scatter)
        # Each ellipse contributes 60 points + 1 None separator → 61 * 3 = 183 entries
        assert trace.x is not None
        assert len(trace.x) == 3 * 61
        assert None in trace.x

    def test_add_pareto_traces_2d_without_noise_produces_traces(self) -> None:
        """_add_pareto_traces_2d without noise adds exactly 3 traces (init, bayes, pareto)."""
        import plotly.graph_objects as go
        objective_values = torch.rand(20, 3)
        figure = go.Figure()
        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=0,
            objective_index_y=1,
            objective_names=['A', 'B', 'C'],
            pareto_optimal_indices=[0, 1, 2],
            n_initial_points=10,
            noise_std_per_objective=None,
        )
        # 3 standard traces: initial points, bayesian points, dominating points
        assert len(figure.data) == 3

    def test_add_pareto_traces_2d_with_ellipse_noise_adds_extra_trace(self) -> None:
        """With noise and ellipse style, one extra trace (the ellipse group) is prepended."""
        import plotly.graph_objects as go
        objective_values = torch.rand(20, 3)
        noise_std = torch.tensor([0.1, 0.1, 0.05])
        figure = go.Figure()
        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=0,
            objective_index_y=1,
            objective_names=['A', 'B', 'C'],
            pareto_optimal_indices=[0, 1, 2],
            n_initial_points=10,
            noise_std_per_objective=noise_std,
            uncertainty_style='ellipse',
        )
        # 3 standard + 3 ellipse traces (initial non-pareto, bayesian non-pareto, dominating)
        assert len(figure.data) == 6
        trace_names = [t.name for t in figure.data]
        assert 'Noise (±1σ)' in trace_names

    def test_add_pareto_traces_2d_with_error_bars_noise_no_extra_trace(self) -> None:
        """With error_bars style, no ellipse trace is added; error_y is set on scatter traces."""
        import plotly.graph_objects as go
        objective_values = torch.rand(20, 3)
        noise_std = torch.tensor([0.1, 0.1, 0.05])
        figure = go.Figure()
        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=0,
            objective_index_y=1,
            objective_names=['A', 'B', 'C'],
            pareto_optimal_indices=[0, 1, 2],
            n_initial_points=10,
            noise_std_per_objective=noise_std,
            uncertainty_style='error_bars',
        )
        # 3 traces, no extra ellipse trace
        assert len(figure.data) == 3
        # The first scatter trace (initial points) should have error_y set
        scatter_traces = [t for t in figure.data if hasattr(t, 'error_y')]
        assert any(t.error_y is not None for t in scatter_traces)

    def test_plot_pareto_front_grid_wires_noise_from_optimiser(self) -> None:
        """plot_pareto_front_grid passes noise_std_per_objective when optimiser has noise_std."""
        optimiser = _make_noisy_vehicle_safety_optimiser()
        figure = plot_pareto_front_grid(optimiser=optimiser)  # type: ignore[arg-type]
        assert figure is not None
        # There should be at least one trace named 'Noise (±1σ)' in the figure
        trace_names = [t.name for t in figure.data]
        assert 'Noise (±1σ)' in trace_names

    def test_plot_pareto_front_wires_noise_from_optimiser(self) -> None:
        """plot_pareto_front passes noise_std_per_objective when optimiser has noise_std."""
        optimiser = _make_noisy_vehicle_safety_optimiser()
        figure = plot_pareto_front(
            optimiser=optimiser,  # type: ignore[arg-type]
            plotted_objective_indices=[0, 1],
        )
        assert figure is not None
        trace_names = [t.name for t in figure.data]
        assert 'Noise (±1σ)' in trace_names

    def test_plot_pareto_front_grid_no_noise_no_ellipse_traces(self) -> None:
        """Without noise_std on the objective, no ellipse trace appears in the figure."""
        objective = VehicleSafety()
        optimiser = bayesian_optimiser(
            n_initial_points=16,
            n_bayesian_points=8,
            n_evaluations_per_step=4,
            objective=objective,
            verbose=False,
            acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 50}},
            model={'training_settings': {'max_iter': 5}},
        )
        for _ in range(4):
            optimiser.run_optimisation_step()
        figure = plot_pareto_front_grid(optimiser=optimiser)
        trace_names = [t.name for t in figure.data]
        assert 'Noise (±1σ)' not in trace_names

    def test_noisy_pareto_dominance_keeps_more_points_than_noiseless(self) -> None:
        """With noise, uncertain points near the boundary should not be discarded.

        We construct a case where point B strictly dominates point A in noiseless
        terms, but the margin is small enough that with noise the dominance is not
        *certain*.  The noisy criterion should keep both points; the noiseless
        criterion should keep only B.
        """
        from veropt.optimiser.optimiser_utility import get_pareto_optimal_points

        # Two points, two objectives (maximisation).
        # B is just slightly better than A on both axes — strictly dominated noiseless.
        a = torch.tensor([[1.0, 1.0]])
        b = torch.tensor([[1.05, 1.05]])
        objective_values = torch.cat([a, b], dim=0)
        variable_values = torch.zeros(2, 1)

        noiseless_result = get_pareto_optimal_points(
            variable_values=variable_values,
            objective_values=objective_values,
        )
        # Noiseless: only B should survive (A is dominated by B).
        # Note: when only 1 point survives, tolist() returns a bare int, not a list.
        noiseless_indices = noiseless_result['index']
        noiseless_indices_list = [noiseless_indices] if isinstance(noiseless_indices, int) else noiseless_indices
        assert len(noiseless_indices_list) == 1
        assert 1 in noiseless_indices_list

        # With noise_std=0.1, the 0.05 gap is well within 2σ=0.2 — A should survive too
        noise_std = torch.tensor([0.1, 0.1])
        noisy_result = get_pareto_optimal_points(
            variable_values=variable_values,
            objective_values=objective_values,
            noise_std_per_objective=noise_std,
        )
        assert len(noisy_result['index']) == 2

    def test_plot_pareto_front_error_bars_style(self) -> None:
        """error_bars style is threaded correctly through plot_pareto_front."""
        optimiser = _make_noisy_vehicle_safety_optimiser()
        figure = plot_pareto_front(
            optimiser=optimiser,  # type: ignore[arg-type]
            plotted_objective_indices=[0, 1],
            uncertainty_style='error_bars',
        )
        assert figure is not None
        # error_bars style: no ellipse trace, but scatter traces should have error bars
        trace_names = [t.name for t in figure.data]
        assert 'Noise (±1σ)' not in trace_names

