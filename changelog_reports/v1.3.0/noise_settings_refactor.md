# Noise Settings Refactor (Schema v1 → v2)

## Overview

`noise`, `noise_lower_bound`, and `train_noise` were previously repeated fields inside
every kernel's `Parameters` dataclass (6 copies). They have been moved out of kernel
settings and into a standalone `noise_settings` sibling, owned by the `GPyTorchSingleModel`
base class. All saved JSON files are now stamped with `schema_version: 2`.

A subsequent bug fix (see **Schema v3** section at the bottom) introduced a schema v3
migration to fix JSON files where GP training inputs were saved with an extra batch
dimension, causing reload crashes.

---

## 1. New types: `NoiseSettingsInputDict` and `NoiseParameters`

**File:** `veropt/optimiser/model.py`, lines 102–115

```python
class NoiseSettingsInputDict(TypedDict, total=False):
    noise: float
    noise_lower_bound: float
    train_noise: bool

@dataclass
class NoiseParameters(SavableDataClass):
    noise: float = 1e-8
    noise_lower_bound: float = 1e-8
    train_noise: bool = False
```

These are the only place where noise defaults and types are declared. The `frozenset`
`_NOISE_KEYS` (line 115) is used by migration and legacy-load detection.

---

## 2. `GPyTorchSingleModel` base class owns noise

**File:** `veropt/optimiser/model.py`, lines 119–252

### 2a. `__init__` — replaced `train_noise: bool` with `noise_settings`

**Before:**
```python
def __init__(self, ..., train_noise: bool = False):
    ...
    self.train_noise = train_noise
```

**After (line 128):**
```python
def __init__(self, ..., noise_settings: Optional[NoiseSettingsInputDict] = None):
    ...
    self._noise_settings = NoiseParameters(**(noise_settings or {}))
    self.train_noise = self._noise_settings.train_noise  # preserved for existing callers
```

### 2b. New concrete method: `_set_up_noise_constraints` (line 249)

Replaces the 5 identical `set_noise` / `set_noise_constraint` blocks that each kernel
had in `_set_up_model_constraints()`:

```python
def _set_up_noise_constraints(self) -> None:
    """Set noise value and lower-bound constraint on the likelihood."""
    self.set_noise(noise=self._noise_settings.noise)
    self.set_noise_constraint(lower_bound=self._noise_settings.noise_lower_bound)
```

Each kernel's `_set_up_model_constraints` now ends with `self._set_up_noise_constraints()`.

### 2c. `gather_dicts_to_save` — noise at state level (line 288)

**Before (v1 JSON):**
```json
"state": {
  "settings": { "lengthscale_lower_bound": 0.1, ..., "noise": 1e-8, "train_noise": false },
  "state_dict": { ... }
}
```

**After (v2 JSON):**
```json
"state": {
  "settings": { "lengthscale_lower_bound": 0.1, ... },
  "noise": 1e-8,
  "noise_lower_bound": 1e-8,
  "train_noise": false,
  "state_dict": { ... }
}
```

### 2d. `from_saved_state` — reads noise from state top-level (line 172)

Noise fields are now read directly from the state dict (v2 format). Passing a v1-format
state dict will raise a `KeyError` — the schema gate in `load_optimiser_from_state` is
the only supported upgrade path:

```python
noise_settings: NoiseSettingsInputDict = {
    'noise': saved_state['noise'],                # KeyError for v1 format — intentional
    'noise_lower_bound': saved_state['noise_lower_bound'],
    'train_noise': saved_state['train_noise'],
}
```

There is no silent fallback or warning here. The responsibility for detecting and
migrating old files sits entirely in `load_optimiser_from_state` (see §6).

---

## 3. Kernel classes — noise stripped, `noise_settings` threaded through

**File:** `veropt/optimiser/kernels.py`

All 6 kernel classes (`MaternKernel`, `DoubleMaternKernel`, `RationalQuadraticKernel`,
`RationalQuadraticMaternKernel`, `SpectralMixtureKernel`, `SpectralDeltaKernel`) changed
identically. Example using `MaternKernel`:

### 3a. TypedDict and dataclass — noise fields removed

**Before:**
```python
class MaternParametersInputDict(TypedDict, total=False):
    lengthscale_lower_bound: float
    lengthscale_upper_bound: float
    nu: float
    noise: float            # removed
    noise_lower_bound: float  # removed
    train_noise: bool         # removed
```

**After:** only the kernel-structural fields remain.

### 3b. `__init__` — new `noise_settings` param before `**settings`

```python
def __init__(
        self,
        n_variables: int,
        noise_settings: Optional[NoiseSettingsInputDict] = None,  # new
        **settings: Unpack[MaternParametersInputDict]
):
    self.settings = MaternParameters(**settings)
    ...
    super().__init__(..., noise_settings=noise_settings)  # was: train_noise=self.settings.train_noise
```

Because `noise_settings` is an explicit keyword argument before `**settings`, existing
callers like `MaternKernel(n_variables=4, lengthscale_upper_bound=2.0)` are unaffected.

### 3c. `from_n_variables_and_settings` — `noise_settings` added (same pattern per kernel)

```python
@classmethod
def from_n_variables_and_settings(
        cls,
        n_variables: int,
        settings: Mapping[str, Any],
        noise_settings: Optional[NoiseSettingsInputDict] = None   # new
) -> 'MaternKernel':
    _validate_typed_dict(settings, MaternParametersInputDict, cls.name)
    return cls(n_variables=n_variables, noise_settings=noise_settings, **settings)
```

### 3d. `_set_up_model_constraints` — noise calls replaced

**Before (one of 5 duplicates):**
```python
def _set_up_model_constraints(self) -> None:
    self.change_lengthscale_constraints(...)
    self.set_noise(noise=self.settings.noise)                          # removed
    self.set_noise_constraint(lower_bound=self.settings.noise_lower_bound)  # removed
```

**After:**
```python
def _set_up_model_constraints(self) -> None:
    self.change_lengthscale_constraints(...)
    self._set_up_noise_constraints()   # single call to base-class method
```

---

## 4. `constructors.py` — `noise_settings` as a sibling to `kernel_settings`

**File:** `veropt/optimiser/constructors.py`, lines 41–44, 171–340

### 4a. `GPytorchModelChoice` TypedDict (line 41)

```python
class GPytorchModelChoice(TypedDict, total=False):
    kernels: ...
    kernel_settings: Optional['KernelInputDict']
    noise_settings: Optional[Union[NoiseSettingsInputDict, list[NoiseSettingsInputDict]]]  # new
    ...
```

### 4b. Three supported calling conventions (line 204)

```python
# 1. Single kernel string + single noise → same noise for all objectives
bayesian_optimiser(..., model={'noise_settings': {'noise': 1e-4}})

# 2. List of kernels + single noise → different kernels, same noise
bayesian_optimiser(..., model={
    'kernels': ['matern', 'double_matern'],
    'noise_settings': {'noise': 1e-4}
})

# 3. List of kernels + list of noise → fully per-objective
bayesian_optimiser(..., model={
    'kernels': ['matern', 'matern'],
    'noise_settings': [{'noise': 1e-4, 'train_noise': True}, {'noise': 1e-5}]
})
```

The resolution logic in `gpytorch_single_model_list` (lines 217–227) broadcasts a single
dict into a `[None] * n_objectives` list, or validates list length, before building kernels.

### 4c. Passing `noise` inside `kernel_settings` raises immediately

`_validate_typed_dict` checks against the updated `MaternParametersInputDict` (which no
longer has noise fields), so:

```python
# Raises AssertionError: "Option 'noise' not recognised for 'matern'. Expected options: [...]"
gpytorch_model(..., kernel_settings={'noise': 1e-4})
```

---

## 5. `OptimiserSettings` — `allow_automatic_json_updates` flag

**File:** `veropt/optimiser/optimiser_utility.py`, lines 35, 73, 97

```python
class OptimiserSettings:
    def __init__(self, ..., allow_automatic_json_updates: bool = False):
        ...
        self.allow_automatic_json_updates = allow_automatic_json_updates

class OptimiserSettingsInputDict(TypedDict, total=False):
    ...
    allow_automatic_json_updates: bool
```

Usage:
```python
optimiser = bayesian_optimiser(..., allow_automatic_json_updates=True)
```

When this is `True` in a saved JSON, loading a file with an outdated schema will
automatically migrate the file before loading.

---

## 6. Schema versioning and migration

**File:** `veropt/optimiser/optimiser_saver_loader.py`

### 6a. `CURRENT_SCHEMA_VERSION = 2` (line 15)

Old files with no `schema_version` key are treated as v1.

### 6b. `save_to_json` stamps version (line 25)

```python
save_dict = object_to_save.gather_dicts_to_save()
save_dict['schema_version'] = CURRENT_SCHEMA_VERSION   # new
```

### 6c. `load_optimiser_from_state` — gates migration (lines 41–74)

```python
schema_version = saved_dict.get('schema_version', 1)   # missing key → v1

if schema_version < CURRENT_SCHEMA_VERSION:
    stored_flag = saved_dict['optimiser']['settings'].get('allow_automatic_json_updates', False)
    allow_updates = allow_automatic_json_updates if allow_automatic_json_updates is not None else stored_flag

    if allow_updates:
        migrate_json(file_name)    # migrates in-place, creates .bak, reloads
    else:
        raise RuntimeError(
            f"The optimiser JSON at '{file_name}' uses schema version {schema_version}, "
            f"but the current schema version is {CURRENT_SCHEMA_VERSION}. ..."
        )
```

The optional `allow_automatic_json_updates: Optional[bool]` parameter (default `None`) lets
the caller override the flag that is stored inside the JSON:

| Value | Behaviour |
|---|---|
| `None` | Defer to the value stored in the JSON (original behaviour, fully backwards-compatible) |
| `True` | Force migration regardless of what the JSON says |
| `False` | Block migration regardless of what the JSON says |

This is important when a user resumes an experiment whose JSON was saved before the flag
existed (so it defaults to `False`) and they want a one-off migration without editing the
file by hand.

### 6c-bis. Threading through experiment constructors

**Files:** `veropt/interfaces/constructors.py`, `veropt/interfaces/experiment.py`

The override is exposed all the way to the user-facing constructor functions:

```python
# One-off migration when resuming an existing experiment
from veropt.interfaces.constructors import experiment

user_experiment = experiment(
    ...,
    allow_automatic_json_updates=True   # overrides the flag in the JSON
)

# Same when branching to a new version
from veropt.interfaces.constructors import experiment_with_new_version

user_experiment = experiment_with_new_version(
    ...,
    allow_automatic_json_updates=True
)
```

The parameter flows: `experiment()` → `Experiment.continue_if_possible()` →
`Experiment._continue_existing()` → `load_optimiser_from_state()` (and the same path for
`experiment_with_new_version` / `Experiment.continue_with_new_version`).

### 6d. `_migrate_v1_to_v2` (line 76)

Walks `optimiser → predictor → state → model → state → model_dicts` and for each
`model_N` moves `{noise, noise_lower_bound, train_noise}` from `model_N.state.settings`
to `model_N.state`:

```python
for model_key, model_dict in model_dicts.items():
    kernel_settings = model_dict['state'].get('settings', {})
    noise_values   = {k: v for k, v in kernel_settings.items() if k in _NOISE_KEYS}
    clean_settings = {k: v for k, v in kernel_settings.items() if k not in _NOISE_KEYS}
    model_dicts[model_key]['state']['settings'] = clean_settings
    model_dicts[model_key]['state'].update(noise_values)
```

### 6e. `migrate_json` — safe in-place rewrite (line 111)

```python
def migrate_json(file_path: str) -> None:
    # 1. Read the file
    # 2. shutil.copy2(file_path, file_path + '.bak')  — backup BEFORE any write
    # 3. Apply _migrate_v1_to_v2
    # 4. Write migrated dict back to original path
```

The backup is created with `shutil.copy2` which preserves file metadata. The original
is only overwritten after the backup exists on disk. If the write fails, the `.bak`
file is intact.

---

## 7. Tests

| Test file | What it covers |
|---|---|
| `tests/test_kernels.py` | `noise_settings` stored correctly; defaults; applied to likelihood after init; v1-format state raises `KeyError`; `noise` in `kernel_settings` raises |
| `tests/test_constructors.py` | Single / broadcast / per-objective `noise_settings`; wrong-length list raises |
| `tests/test_optimiser_saver_loader.py` | Schema version stamped on save; noise at correct level in JSON; round-trip preserves values; `_migrate_v1_to_v2` unit test; `migrate_json` creates backup and updates file; v1 raises without flag; v1 auto-migrates with flag; no-op on current version |

---

## Migration guide for existing users

**Option A — migrate the file manually (recommended):**
```python
from veropt.optimiser.optimiser_saver_loader import migrate_json
migrate_json('path/to/my_optimiser.json')
# Creates my_optimiser.json.bak, rewrites my_optimiser.json to schema v2
```

**Option B — let veropt migrate automatically on load (flag already in JSON):**

Add `allow_automatic_json_updates=True` to your optimiser config. The next time the optimiser
is saved and then reloaded, any format mismatch will be handled automatically (with a backup).

**Option C — one-off override at the call site (flag not yet in JSON / set to False):**

Pass the override directly to the constructor — no JSON editing needed:
```python
user_experiment = experiment(..., allow_automatic_json_updates=True)
# or
user_experiment = experiment_with_new_version(..., allow_automatic_json_updates=True)
```

After the migration, the next `save_to_json` will stamp the current schema version so the
file is up to date and the override is no longer needed.

**What NOT to do:** passing `noise` / `noise_lower_bound` / `train_noise` inside
`kernel_settings` (either in Python or in a settings JSON) will now raise an error.
Move those fields to `noise_settings` at the same level.

---

## Schema v3 — `train_inputs` tuple unwrap (bug fix)

**File:** `veropt/optimiser/model.py`, `veropt/optimiser/optimiser_saver_loader.py`

### Problem
`GPyTorchSingleModel.gather_dicts_to_save` was saving `model_with_data.train_inputs`
directly. In gpytorch, `train_inputs` is a `tuple` (e.g. `(X_tensor,)`). When serialised
to JSON and reloaded via `torch.tensor(data)`, the tuple wrapper produced shape
`[1, n_points, n_vars]` instead of `[n_points, n_vars]`, giving every reloaded model
`batch_shape = [1]`. BoTorch's `prune_inferior_points_multi_objective` (used inside
`qLogNoisyExpectedHypervolumeImprovement`) then raised:

```
botorch.exceptions.errors.UnsupportedError: Models with multiple batch dims are
currently unsupported by `prune_inferior_points_multi_objective`.
```

This crash only appeared on **reload** — training ran fine on first use.

### Fix
1. **`gather_dicts_to_save`** — unwrap the tuple: save `model_with_data.train_inputs[0]`
   (equivalently `.squeeze(0)` on reload). Also save the live constraint lower bound
   (`physical_variance * 0.99`) so the noise constraint is correctly restored before
   `load_state_dict`.
2. **`prediction.py` — reload path** — removed the `_apply_physical_noise` call on
   `train=False`. The state dict already encodes the correct `raw_noise`, and the
   constraint is now properly restored by `from_saved_state`.
3. **`_migrate_v2_to_v3`** — migrates existing JSONs saved with the tuple-wrapped
   `train_inputs` by squeezing out the extra batch dimension in-place.

### `CURRENT_SCHEMA_VERSION = 3`
Old files (v1 or v2) are automatically migrated in sequence:
v1 → v2 → v3, provided `allow_automatic_json_updates=True`.

