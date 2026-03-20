# AGENTS.md - Guide for AI Coding Agents

## Project Overview

**veropt** is a user-friendly Bayesian Optimization library designed for expensive optimization problems. It's built around PyTorch, GPyTorch, and BoTorch for Gaussian Process modeling and acquisition function optimization.

### Core Workflow
The library follows this optimization loop:
1. **Initial Phase**: Generate and evaluate initial random points
2. **Bayesian Phase**: Build GP surrogate model → optimize acquisition function → evaluate suggested points
3. **Visualization**: Plot predictions, acquisition functions, and evaluated points with Plotly

**Key Maximization Convention**: veropt always maximizes objectives. Minimization requires negating objectives.

## Architecture Overview

### Three Main Modules

#### `veropt/optimiser/` - Core Optimization Engine
- **`optimiser.py`**: `BayesianOptimiser` class - main entry point managing the optimization loop
- **`constructors.py`**: `bayesian_optimiser()` factory function with TypedDict-based configuration
- **`objective.py`**: Abstract `Objective` base class with `CallableObjective` and `InterfaceObjective` variants
- **`model.py`**: GPyTorch-based GP models (`GPyTorchFullModel`, `GPyTorchSingleModel`)
- **`acquisition.py`**: Acquisition functions (qLogEHVI, UCB) using BoTorch
- **`acquisition_optimiser.py`**: Dual annealing optimizer for finding next points
- **`prediction.py`**: `BotorchPredictor` wrapper for posterior sampling
- **`normalisation.py`**: Input/output normalization (StandardNormaliser, RobustNormaliser)
- **`saver_loader_utility.py`**: Serialization via `SavableClass` interface - all core objects implement this

#### `veropt/interfaces/` - External Simulation Integration
- **`experiment.py`**: `Experiment` class - recommended high-level interface orchestrating simulations with batch management and state persistence
- **`simulation.py`**: Abstract `SimulationRunner` for executing objective functions (local or SLURM-based)
- **`batch_manager.py`**: Submit jobs locally or to SLURM clusters
- **`result_processing.py`**: `ResultProcessor` to extract objectives from simulation outputs
- **`experiment_utility.py`**: State management (`ExperimentalState`) and configuration (`ExperimentConfig`)

#### `veropt/graphical/` - Interactive Visualization
- **`visualisation.py`**: Main entry point - creates Plotly dashboards
- **`_model_visualisation.py`**: Plots GP predictions with uncertainty
- **`_pareto_front.py`**: Multi-objective Pareto front visualization

### Data Flow

```
Objective (bounds, n_variables, n_objectives)
    ↓
Normaliser (StandardNormaliser/RobustNormaliser)
    ↓
Predictor (GP surrogate model)
    ↓
AcquisitionFunction (qLogEHVI/UCB)
    ↓
AcquisitionOptimiser (Dual Annealing)
    ↓
Experiment/Simulator (LocalSimulation/SlurmSimulation)
    ↓
ResultProcessor (extract objectives)
    ↓
BayesianOptimiser.run_optimisation_step()
```

## Critical Developer Patterns

### 0. Code Design Priorities
Follow this hierarchy when generating code - **always start minimal**:
1. **KISS (Keep It Simple, Stupid)** - Primary goal. Start with minimal implementation.
2. **YAGNI (You Aren't Gonna Need It)** - Avoid speculation. If edge case not demonstrated, add `assert` instead.
3. **SRP (Single Responsibility Principle)** - One clear purpose per function/class
4. **Rest of SOLID principles**

Example: Don't build a general strategy pattern if a simple `if/else` works. Add assertions for unsupported cases rather than over-engineering.

### 1. Natural Naming Conventions
- **Loops**: Use descriptive names, not single letters. Examples:
  - `for point_number in range(n_points):` not `for i in range(n_points):`
  - `for objective_index, obj_value in enumerate(objectives):` not `for i, val in enumerate(objectives):`
  - `for step in range(max_steps):` not `for i in range(max_steps):`
- **Variables**: Name for clarity even if verbose. `evaluated_objectives_normalised` is better than `evals`.
- **Boolean flags**: Prefix with verb (is_, has_, can_). Example: `should_normalize_inputs` not `normalize`.

### 2. Configuration via TypedDict
The library uses `TypedDict` for flexible, validated configuration dictionaries rather than explicit arguments:

```python
optimiser = bayesian_optimiser(
    n_initial_points=16,
    n_bayesian_points=32,
    n_evaluations_per_step=4,
    objective=objective,
    model={'training_settings': {'max_iter': 50}},  # TypedDict
    acquisition={'parameters': {'beta': 0.2}},       # TypedDict
    acquisition_optimiser={'optimiser': 'dual_annealing'}
)
```

See `optimiser_utility.py` for `OptimiserSettingsInputDict` definition.

### 3. Savable/Loadable Architecture
All core classes inherit from `SavableClass` (abstract base in `saver_loader_utility.py`):
- Implement `gather_dicts_to_save()` to return serializable dict
- Implement `from_saved_state(saved_state: dict)` for deserialization
- For dataclasses, inherit from `SavableDataClass` instead - automatically implements both methods via `asdict()`
- This enables checkpointing: `save_to_json(optimiser, path)` / `load_optimiser_from_state(path)`

### 4. Normalization as Wrapper
Normalisation is transparent - objectives/variables can be in real or normalized space:
- Stored in `_normaliser_variables` and `_normaliser_objectives` (optional)
- Properties like `bounds_normalised`, `evaluated_variables_normalised` cache normalized versions
- Always work in real units internally, normalize only when needed by GP

### 5. Torch as Default Numerics
- `torch.set_default_dtype(torch.float64)` set at module load
- All numeric operations use PyTorch tensors
- `numpy` compatibility handled explicitly (see `check_incoming_objective_dimensions_fix_1d`)

### 6. Decorator Pattern for Input Validation
Functions use `@_check_input_dimensions` decorator (in `acquisition.py`, `model.py`) to:
- Enforce consistent variable/objective dimensions
- Support flexible positional/keyword arguments via `**kwargs`
- See `utility.py` for `enforce_amount_of_positional_arguments`

### 7. Multi-Objective & Visualization Reference Points
- Single-objective: `n_objectives=1`
- Multi-objective: Nadir point automatically generated via `get_nadir_point()` for qLogEHVI acquisition
- **BoTorch Reference Point**: Mathematical entity used internally in acquisition function calculations. Do not confuse with veropt's reference points.
- **veropt Reference Points** (visualization only): User-supplied points for comparison in plots - **distinct from BoTorch's reference point**
  - Contains both `variable_values` and `objective_values` (stored as `torch.Tensor`)
  - Interfaces aim to support automatic evaluation of user variable values to extract objective values for visualization
  - Cannot be auto-created; requires explicit user input

## Testing & Validation

### Run Tests
```bash
cd /lustre/hpc/ocean/aster07/PycharmProjects/veropt
python local_workflows/tests.py
# or directly: pytest
```

### Type Checking
```bash
python local_workflows/type_checking.py
# or: mypy veropt tests examples
```

### Linting
```bash
python local_workflows/linting.py
# or: flake8 . (ignores E402 - see setup.cfg)
```

**Type Checking Rules** (`mypy.ini`):
- `disallow_untyped_defs = True` - strict typing enforced
- `follow_untyped_imports = False` - external libs may be untyped
- `ignore_missing_imports = True` - for torch/gpytorch stubs

### Test Structure
- Test files match source structure: `tests/test_*.py` for `veropt/*.py`
- Use practice objectives from `veropt.optimiser.practice_objectives` for testing: `Hartmann` (3/4/6-d), `VehicleSafety` (5-d, multi-objective), `DTLZ1` (configurable)
- Example: `test_run_optimisation_step()` in `test_optimiser.py` runs 5 steps with reduced iterations

## Examples & Interfaces

### Quick Start
- `examples/example_single_objective.py` - Basic single-objective optimization
- `examples/example_multi_objective.py` - Multi-objective with Pareto front
- `examples/example_visualisation.py` - Interactive dashboard

### External Simulations
- `examples/interfaces/example_local_veros_experiment.py` - Local VEROS ocean model
- `examples/interfaces/example_slurm_veros_simulation.py` - VEROS on SLURM cluster
- Config files in `examples/` show JSON configuration patterns

## Key Files for Different Tasks

| Task | Files |
|------|-------|
| Add acquisition function | `optimiser/acquisition.py`, `optimiser/constructors.py` |
| Add surrogate model | `optimiser/model.py`, `optimiser/prediction.py` |
| Add normalisation | `optimiser/normalisation.py`, `optimiser/optimiser.py` |
| Add simulator interface | `interfaces/simulation.py`, `interfaces/batch_manager.py` |
| Visualization features | `graphical/visualisation.py`, `graphical/_*.py` |
| Configuration schema | `optimiser/constructors.py` (TypedDict definitions) |

## Important Notes for AI Agents

1. **Float64 Requirement**: PyTorch defaults to float32. Always preserve `torch.set_default_dtype(torch.float64)` in `__init__.py`.

2. **Flake8 Exception**: E402 (module import not at top) is ignored in `setup.cfg` because torch dtype must be set before importing submodules.

3. **NaN Handling**: Multi-objective experiments mask NaN objectives with `_mask_nans()` in `experiment.py` - edge case for robust simulations.

4. **Subclass Registry Pattern**: `get_all_subclasses()` in `saver_loader_utility.py` enables polymorphic deserialization by class name - used for all acquisition functions, models, normalizers.

5. **Experiment State Machine**: `ExperimentalState` in `experiment_utility.py` tracks `next_point` index, pending simulations, and results - critical for resuming interrupted optimizations.

6. **Default Settings**: `veropt/optimiser/default_settings.json` contains hardcoded defaults - modify via TypedDict overrides in constructor, not by editing JSON directly.

7. **Pytest + Python 3.13**: Requires `python >3.13` per `setup.py`. Codebase uses modern syntax (PEP 695 type aliases with `type Foo = ...`).

