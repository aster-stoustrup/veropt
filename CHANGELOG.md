# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.3.0] - 2026-04-21

### Added
- **Observation noise support (V1)**: constant per-objective noise std can now be set on any
  `Objective` via `noise_std: dict[str, float]`. The noise is pinned in the GP likelihood,
  automatically selected as `qLogNoisyEHVI` for multi-objective problems, and shown as
  uncertainty ellipses (or error bars) on Pareto front plots.
  See `changelog_reports/v1.3.0/noise_v1_implementation.md` for full details.
- **Noise-aware Pareto front**: `get_pareto_optimal_points` now accepts
  `noise_std_per_objective` and uses the Fieldsend & Everson (2015) "certain dominance"
  criterion. `plot_pareto_front` / `plot_pareto_front_grid` pass noise through automatically
  and accept `uncertainty_style='ellipse'` (default) or `'error_bars'`.
- **Noisy Pareto front example**: `examples/example_noisy_pareto_front.py`.
- **Experiment rollback utility**: `veropt.interfaces.rollback.rollback_experiment` rolls
  an experiment back to a previous batch boundary, restoring JSON state and optionally
  renaming or deleting simulation result folders. CLI available via
  `veropt.interfaces.cli`. See `changelog_reports/v1.3.0/rollback_implementation.md`.
- **Schema versioning and auto-migration**: optimiser JSON files are now stamped with
  `schema_version`. Migrations v1→v2→v3 are applied automatically when
  `allow_automatic_json_updates=True` (with backup). The flag can be overridden at the
  call site via `experiment(..., allow_automatic_json_updates=True)` without editing
  the JSON. See `changelog_reports/v1.3.0/noise_settings_refactor.md`.
- **Normaliser `transform_scale` / `inverse_transform_scale`**: new methods that apply
  only the scale part of normalisation (no mean shift), used for transforming noise std
  into model space.
- **`NoiseSettingsInputDict` TypedDict**: explicit, validated noise configuration as a
  sibling to `kernel_settings` in the model constructor. Supports single, broadcast, and
  per-objective noise settings.
- **String-based point selection in prediction plots**: `plot_prediction_grid`,
  `plot_prediction_surface`, and `plot_prediction_surface_grid` now accept string
  selectors for `evaluated_point`: `"best"`, `"best {objective_name}"`, `"suggested N"`.
  `evaluated_point` on `plot_prediction_surface` is now optional (defaults to `None`).
  See `changelog_reports/v1.3.0/visualisation_improvements.md`.
- **`allow_automatic_json_updates` exposed to experiment constructors**: pass the flag
  directly to `experiment()` / `experiment_with_new_version()` to trigger a one-off JSON
  migration without editing the file by hand.
- **Fake SLURM batch manager for testing**: `MockBatchManager` in
  `tests/interfaces/test_slurm_experiment.py` enables integration tests of the experiment
  submission loop without a real SLURM queue.
- **`noise_std` in `ExperimentConfig`**: noise is now configurable via the experiment
  config JSON, keyed by objective name. No changes to the optimiser settings JSON needed.

### Changed
- **Kernel noise refactor**: `noise`, `noise_lower_bound`, and `train_noise` have been
  moved out of each kernel's settings dataclass and into a new `_noise_settings`
  (`NoiseParameters`) owned by the `GPyTorchSingleModel` base class.
  Passing these keys inside `kernel_settings` now raises immediately.
- **Default plot evaluation point**: `choose_plot_point` now defaults to the **best
  evaluated point** (highest weighted objective sum) instead of the first suggested point.
- **`_set_up_model_constraints` refactored**: the base class now owns
  `_set_up_noise_constraints()`, called by all kernels — eliminating 5 duplicate blocks.
- **`CURRENT_SCHEMA_VERSION = 3`** (was 2 after the kernel noise refactor).

### Fixed
- **NumPy 2.4 compatibility** (`TypeError: only 0-dimensional arrays can be converted to
  Python scalars`): `ProximityPunishmentSequentialOptimiser._sample_acq_func` used
  `.detach().numpy()` when assigning to a numpy scalar slot. Fixed with `.detach().item()`.
  (closes issue #22)
- **Noisy multi-objective reload crash** (`UnsupportedError: Models with multiple batch
  dims`): `gather_dicts_to_save` was saving `model_with_data.train_inputs` as a tuple,
  producing shape `[1, n_points, n_vars]` on reload and `batch_shape=[1]`. Fixed by
  unwrapping the tuple on save. Schema v3 migration fixes existing JSON files.

## [1.2.0] - 10-12-2025
Major improvements of visual tools and added new kernels.

### Added
- New built-in kernels
- 3d prediction plot
- Ability to run all graphs without normalisation (now default)
- Template for writing a new experiment
- Two visualisation examples
- General improvement to most visual tools
- 

### Changed
- Moved some internal visual methods to their own folders
  - 'visualisation.py' is now where users should go to find visual methods
- Changed naming of public visual methods
- Suggested points are now reset after loading new data instead of 
  after saving suggestions
- Learning rate setting has been fixed and is now
  - a) functional and
  - b) residing in the model optimiser where it belongs
- Model optimiser has been cleaned up and now follows same system as similar objects
- Fixed issue from pydantic with saving nan's to json
- jsons are pretty-printed
- Objective values will not be re-calculated if they're already in exp state
- Fixed minor bug when saving suggested steps

## [1.1.2] - 17-11-2025
Added the ability to use existing run with a new objective

### Added
- New experiment constructor that will use new ability to create 
  new version of existing experiment.

### Changed
- Name and location of optimiser and experiment state jsons

## [1.1.0] - 25-10-2025
Updated interfaces to allow pausing and resuming runs.

### Added
- Ability to resume runs that have been stopped
- Support for optimiser configuration
- Experiment will save optimiser state and reload it

### Changed
- Some internal refactoring on the Experiment class

## [1.0.0] - 15-07-2025
Refactor of the entire project! 

### Added
- New interfaces folder for setting up optimisation problems on e.g. slurm
- It is possible to save the optimiser again, now in a readable, stable json file
- New setting file (also json) where optimiser configuration can be saved
- New constructor functions that can be called instead of creating classes directly
- veropt is now typed and checked by mypy

### Changed
- Internal structure
- Interfaces
- Examples

### Removed
- The GUI is not currently available but will hopefully return in the future

## [0.6.0] - 28-02-2025
### Added
- Changelog :))
- New visualisation tools in plotly
- Test folder!
  - First tests added for normalisation, more will follow in veropt 1.0

### Changed
- Dependencies are updated to newest versions
- Normalisation
  - Should work more correctly and be more robust now
- Sequence optimiser (proximity punish)
  - Now measuring scale globally to remove assumption of acq func range from 0 to a positive number
  - Furthermore checking for multiple disjoint distributions of acquisition function values and (if found) uses std of 
    top distribution
  - All this should ensure correct behaviour and avoid bugs that caused optimisation to 1) choose the same point 
    multiple times or 2) making the punishment dominate the landscape

### Removed
- Temporary:
  - Saver (will come back in later release!)
- Possibly permanent:
  - UCB with noise