# Bug Fixes

---

## NumPy 2.4 compatibility — `ProximityPunishmentSequentialOptimiser`

**File:** `veropt/optimiser/acquisition_optimiser.py`, line ~478
**Closes:** [issue #22](https://github.com/aster-stoustrup/veropt/issues/22)

### Symptom
Running with NumPy ≥ 2.4 raised:

```
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

in `ProximityPunishmentSequentialOptimiser._sample_acq_func()`.

### Root cause
NumPy 2.4 tightened the rule that assigning to a scalar slot of a numpy array
(`samples[coord_ind] = value`) requires `value` to be 0-dimensional.
`sample.detach().numpy()` returned a shape-`[1]` array — no longer accepted.

### Fix
```python
# Before
samples[coord_ind] = sample.detach().numpy()

# After
samples[coord_ind] = sample.detach().item()
```

`.item()` extracts a plain Python float, which is always a valid assignment target
in numpy regardless of version. It also implicitly detaches from the computation
graph, so the memory-leak protection is preserved.

---

## Noisy multi-objective reload crash — `batch_shape=[1]`

**File:** `veropt/optimiser/model.py`, `veropt/optimiser/prediction.py`
**Full details:** see `noise_settings_refactor.md` → *Schema v3* section.

### Symptom
Reloading a saved noisy multi-objective optimiser raised:

```
botorch.exceptions.errors.UnsupportedError: Models with multiple batch dims are
currently unsupported by `prune_inferior_points_multi_objective`.
```

The optimiser ran fine during training — the crash only appeared on reload.

### Root cause
`gather_dicts_to_save` saved `model_with_data.train_inputs` (a gpytorch tuple),
which on reload produced tensors with shape `[1, n_points, n_vars]` instead of
`[n_points, n_vars]`, giving `batch_shape = [1]`.

### Fix
- Unwrap the tuple on save: `model_with_data.train_inputs[0]` (→ correct shape).
- Also save the live noise constraint lower bound so the pinned noise is correctly
  restored before `load_state_dict`.
- Removed the `_apply_physical_noise` call on the reload path — the state dict
  already encodes the correct `raw_noise`.
- Added schema v3 migration (`_migrate_v2_to_v3`) to fix existing JSON files.
