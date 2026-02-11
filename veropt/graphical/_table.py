from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Literal

import plotly.graph_objects as go
import torch
from plotly import colors

from veropt.graphical._visualisation_utility import get_continuous_colour
from veropt.optimiser.optimiser_utility import _format_number


def _build_table(
        chosen_points: list[int],
        variable_names: list[str],
        objective_names: list[str],
        evaluated_variable_values: torch.Tensor,
        evaluated_objective_values: torch.Tensor,
        reference_variable_values: Optional[torch.Tensor] = None,
        reference_objective_values: Optional[torch.Tensor] = None,
) -> dict[Literal['variables', 'objectives'], list[dict[str, Optional[float]]]]:

    variable_rows = []
    for variable_index, variable_name in enumerate(variable_names):
        row = {"variable": variable_name}

        if reference_variable_values is not None:
            row["default"] = float(reference_variable_values[variable_index])

        for point_number in chosen_points:
            row[f"point_{point_number}"] = float(evaluated_variable_values[point_number, variable_index])

        variable_rows.append(row)

    objective_rows = []
    for objective_index, objective_name in enumerate(objective_names):
        row = {"objective": objective_name}

        if reference_objective_values is not None:
            row["default"] = float(reference_objective_values[objective_index])

        for point_number in chosen_points:
            row[f"point_{point_number}"] = float(evaluated_objective_values[point_number, objective_index])

        objective_rows.append(row)

    return {
        'variables': variable_rows,
        'objectives': objective_rows
    }


def _build_cell_values(
        rows: list[dict[str, Optional[float]]],
        columns: list[str],
        data_name: str
) -> list[list]:
    """
    Build cell values for a table from rows and column names.

    Args:
        rows: List of dicts containing row data.
        columns: List of column names.

    Returns:
        List of cell value lists, one per column.
    """
    cell_values = [[] for _ in range(len(columns))]

    for row in rows:
        cell_values[0].append(row[data_name])
        if "Default" in columns:
            cell_values[1].append(row.get("default", ""))
        for col_idx, col in enumerate(columns[2:] if "Default" in columns else columns[1:]):
            actual_col_idx = col_idx + (2 if "Default" in columns else 1)
            cell_values[actual_col_idx].append(row.get(col, ""))

    return cell_values


from plotly.express import colors
from veropt.graphical._visualisation_utility import get_continuous_colour


def _calculate_cell_colors(
        variable_rows: list[list[Optional[float]]],
        bounds: torch.Tensor,
        columns: list[str],
        reference_values: Optional[torch.Tensor] = None,
        color_scale: Optional[list[list]] = None
) -> list[list[str]]:

    if color_scale is None:
        color_scale = colors.get_colorscale("RdBu")

    n_variables = bounds.shape[1]
    variable_colors: list[list[str]] = []

    for row in variable_rows:
        row_colors: list[str] = []
        if isinstance(row[0], str):
            pass
        else:
            for col_idx, value in enumerate(row):
                if col_idx == 0 or value is None:
                    # First column is point name, use white
                    row_colors.append('white')
                else:
                    var_idx = col_idx - 1  # Adjust for point name column
                    if var_idx < n_variables:
                        lower_bound = float(bounds[0, var_idx])
                        upper_bound = float(bounds[1, var_idx])
                        bound_range = upper_bound - lower_bound

                        if bound_range == 0:
                            row_colors.append('white')
                        else:
                            if reference_values is not None:
                                ref_value = float(reference_values[var_idx])
                                # Distance from reference to each bound
                                dist_to_lower = abs(ref_value - lower_bound)
                                dist_to_upper = abs(ref_value - upper_bound)
                                max_dist = max(dist_to_lower, dist_to_upper)

                                # Distance from value to reference
                                dist_from_ref = abs(value - ref_value)

                                # Normalize: max color at the bound furthest from reference
                                normalized_value = dist_from_ref / max_dist if max_dist > 0 else 0.0
                            else:
                                # Fallback: normalize within bounds (0 at lower, 1 at upper)
                                normalized_value = (value - lower_bound) / bound_range

                            normalized_value = max(0.0, min(1.0, normalized_value))
                            color = get_continuous_colour(color_scale, normalized_value)
                            row_colors.append(color)
                    else:
                        row_colors.append('white')
            variable_colors.append(row_colors)

    return variable_colors



def _plot_table(
        table_data: dict[Literal['variables', 'objectives'], list[dict[str, Optional[float]]]],
        bounds: torch.Tensor,
        reference_values: Optional[torch.Tensor] = None,
        color_scale: Optional[list[list]] = None
) -> go.Figure:
    variable_rows = table_data['variables']
    objective_rows = table_data['objectives']

    # Extract column names from parameters
    columns = ["Parameter"]
    if any("default" in row for row in variable_rows):
        columns.append("Default")
    columns.extend([key for key in variable_rows[0].keys() if key.startswith("point_")])

    # Build cell values for both sections
    variable_cell_values = _build_cell_values(
        rows=variable_rows,
        columns=columns,
        data_name='variable'
    )
    objective_cell_values = _build_cell_values(
        rows=objective_rows,
        columns=columns,
        data_name='objective'
    )

    # Combine with separator row
    cell_values = [[] for _ in range(len(columns))]
    for col_idx in range(len(columns)):
        cell_values[col_idx].extend(variable_cell_values[col_idx])
        cell_values[col_idx].append("Objectives" if col_idx == 0 else "")
        cell_values[col_idx].extend(objective_cell_values[col_idx])

    # Format floats to 3 decimal places
    for col_idx in range(len(columns)):
        for row_idx in range(len(cell_values[col_idx])):
            value = cell_values[col_idx][row_idx]
            if isinstance(value, float):
                cell_values[col_idx][row_idx] = _format_number(value)

    # Calculate colors for variable rows
    variable_colors = _calculate_cell_colors(
        variable_rows=variable_cell_values,
        bounds=bounds,
        columns=columns,
        reference_values=reference_values,
        color_scale=color_scale
    )

    # Create fill colors with separator and objective rows
    separator_row_index = len(variable_rows)
    n_rows = len(cell_values[0])

    fill_colors = [[] for _ in range(len(columns))]
    for col_idx in range(len(columns)):
        for row_idx in range(n_rows):
            if row_idx < len(variable_rows):
                fill_colors[col_idx].append(variable_colors[col_idx][row_idx])
            elif row_idx == separator_row_index:
                fill_colors[col_idx].append('rgb(200, 212, 227)')
            else:
                fill_colors[col_idx].append('rgb(235, 240, 248)')

    figure = go.Figure(data=[go.Table(
        header=dict(values=columns),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors
        )
    )])

    return figure


def _plot_bounds_table(
        variable_names: list[str],
        bounds: torch.Tensor
) -> go.Table:
    """
    Create a table displaying the bounds of all variables.

    Args:
        variable_names: List of variable names.
        bounds: Tensor containing bounds with shape [2, n_variables].

    Returns:
        A plotly Table object showing variable bounds.
    """
    bounds_columns = ["Variable", "Lower Bound", "Upper Bound"]
    bounds_cell_values = [
        variable_names,
        [_format_number(float(bounds[0, i])) for i in range(len(variable_names))],
        [_format_number(float(bounds[1, i])) for i in range(len(variable_names))]
    ]

    return go.Table(
        header=dict(values=bounds_columns),
        cells=dict(values=bounds_cell_values)
    )



def _save_table_as_csv(
        table_data: dict[Literal['variables', 'objectives'], list[dict[str, Optional[float]]]],
        filepath: str | Path
) -> None:
    """
    Save table data to a CSV file.

    Args:
        table_data: Dict with 'variables' and 'objectives' keys containing row data.
        filepath: Path where the CSV file will be saved.
    """
    variable_rows = table_data['variables']
    objective_rows = table_data['objectives']

    # Extract column names from variable rows
    columns = ["Name"]
    if any("default" in row for row in variable_rows):
        columns.append("Default")
    columns.extend([key for key in variable_rows[0].keys() if key.startswith("point_")])

    with open(filepath, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()

        # Write variable rows
        for row in variable_rows:
            csv_row = {"Name": row["variable"]}
            if "default" in row:
                csv_row["Default"] = row["default"]
            for key in row.keys():
                if key.startswith("point_"):
                    csv_row[key] = row[key]
            writer.writerow(csv_row)

        # Write objectives header and rows
        writer.writerow({"Name": "Objectives"})
        for row in objective_rows:
            csv_row = {"Name": row["objective"]}
            if "default" in row:
                csv_row["Default"] = row["default"]
            for key in row.keys():
                if key.startswith("point_"):
                    csv_row[key] = row[key]
            writer.writerow(csv_row)
