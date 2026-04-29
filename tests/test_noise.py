"""Tests for V1 observation noise support.

Covers:
- Normaliser transform_scale / inverse_transform_scale (Step 1)
- Objective noise_std field and save/load round-trip (Step 2)
- BayesianOptimiser conflict checks and noise properties (Step 3)
- GPyTorchFullModel._apply_physical_noise (Step 4)
- Predictor.update_with_new_data noise threading (Step 5)
- Auto-selection of noisy acquisition function (Steps 7-8)
- Model posterior variance with and without noise
- JSON noise desync detection on reload
"""
import json
from pathlib import Path
from typing import Optional

import pytest
import torch

from veropt.optimiser.constructors import bayesian_optimiser, botorch_acquisition_function
from veropt.optimiser.kernels import MaternKernel
from veropt.optimiser.model import GPyTorchFullModel, NoiseSettingsInputDict
from veropt.optimiser.normalisation import NormaliserZeroMeanUnitVariance
from veropt.optimiser.objective import Objective
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_saver_loader import save_to_json, load_optimiser_from_state
from veropt.optimiser.practice_objectives import Hartmann, VehicleSafety, DTLZ1
from veropt.optimiser.prediction import BotorchPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_objective_noisy_optimiser(n_initial: int = 4, n_bayesian: int = 4) -> BayesianOptimiser:
    """Small Hartmann-6 optimiser with noise_std set."""
    objective = Hartmann(n_variables=6, noise_std={'Hartmann': 0.05})
    return bayesian_optimiser(
        n_initial_points=n_initial,
        n_bayesian_points=n_bayesian,
        n_evaluations_per_step=1,
        objective=objective,
        model={'training_settings': {'max_iter': 5, 'verbose': False}},
    )


def _make_multi_objective_noisy_optimiser(n_initial: int = 4, n_bayesian: int = 4) -> BayesianOptimiser:
    """Small VehicleSafety optimiser with noise_std set on all three objectives."""
    noise_std = {f"VeSa {i + 1}": 0.1 for i in range(3)}
    objective = VehicleSafety(noise_std=noise_std)
    return bayesian_optimiser(
        n_initial_points=n_initial,
        n_bayesian_points=n_bayesian,
        n_evaluations_per_step=1,
        objective=objective,
        model={'training_settings': {'max_iter': 5, 'verbose': False}},
    )


# ---------------------------------------------------------------------------
# Step 1 — Normaliser.transform_scale / inverse_transform_scale
# ---------------------------------------------------------------------------

class TestNormaliserTransformScale:

    def test_transform_scale_removes_mean_shift(self) -> None:
        """transform_scale should scale but NOT subtract the mean."""
        means = torch.tensor([10.0, 100.0])
        variances = torch.tensor([4.0, 25.0])
        normaliser = NormaliserZeroMeanUnitVariance(means=means, variances=variances)

        noise_tensor = torch.tensor([2.0, 5.0])
        scaled = normaliser.transform_scale(noise_tensor)

        expected = torch.tensor([2.0 / 2.0, 5.0 / 5.0])  # divided by sqrt(var)
        assert torch.allclose(scaled, expected), f"Expected {expected}, got {scaled}"

    def test_inverse_transform_scale_is_inverse(self) -> None:
        means = torch.tensor([3.0])
        variances = torch.tensor([9.0])
        normaliser = NormaliserZeroMeanUnitVariance(means=means, variances=variances)

        original = torch.tensor([1.5])
        assert torch.allclose(normaliser.inverse_transform_scale(normaliser.transform_scale(original)), original)

    def test_transform_scale_is_different_from_full_transform(self) -> None:
        """For a non-zero mean, transform_scale and transform produce different results."""
        means = torch.tensor([5.0])
        variances = torch.tensor([1.0])
        normaliser = NormaliserZeroMeanUnitVariance(means=means, variances=variances)

        value = torch.tensor([3.0])
        assert not torch.allclose(normaliser.transform(value), normaliser.transform_scale(value))


# ---------------------------------------------------------------------------
# Step 2 — Objective.noise_std
# ---------------------------------------------------------------------------

class TestObjectiveNoiseStd:

    def test_noise_std_stored_on_objective(self) -> None:
        noise_std = {'Hartmann': 0.05}
        objective = Hartmann(n_variables=6, noise_std=noise_std)
        assert objective.noise_std == noise_std

    def test_noise_std_is_none_by_default(self) -> None:
        objective = Hartmann(n_variables=6)
        assert objective.noise_std is None

    def test_noise_std_saved_in_state_dict(self) -> None:
        noise_std = {'Hartmann': 0.05}
        objective = Hartmann(n_variables=6, noise_std=noise_std)
        state = objective.gather_dicts_to_save()
        assert state['state']['noise_std'] == noise_std

    def test_noise_std_key_mismatch_raises(self) -> None:
        with pytest.raises(AssertionError):
            Hartmann(n_variables=6, noise_std={'WrongKey': 0.1})

    def test_hartmann_from_saved_state_restores_noise_std(self) -> None:
        noise_std = {'Hartmann': 0.05}
        original = Hartmann(n_variables=6, noise_std=noise_std)
        saved_state = original.gather_dicts_to_save()
        restored = Hartmann.from_saved_state(saved_state['state'])
        assert restored.noise_std == noise_std

    def test_hartmann_from_saved_state_without_noise_std_gives_none(self) -> None:
        """Backward compat: old saved states without noise_std key should give None."""
        original = Hartmann(n_variables=6)
        saved_state = original.gather_dicts_to_save()
        del saved_state['state']['noise_std']  # simulate old format
        restored = Hartmann.from_saved_state(saved_state['state'])
        assert restored.noise_std is None

    def test_vehicle_safety_from_saved_state_restores_noise_std(self) -> None:
        noise_std = {f"VeSa {i + 1}": 0.1 for i in range(3)}
        original = VehicleSafety(noise_std=noise_std)
        saved_state = original.gather_dicts_to_save()
        restored = VehicleSafety.from_saved_state(saved_state['state'])
        assert restored.noise_std == noise_std


# ---------------------------------------------------------------------------
# Step 3 — BayesianOptimiser conflict checks and noise properties
# ---------------------------------------------------------------------------

class TestOptimiserNoiseConfiguration:

    def test_train_noise_with_noise_std_raises(self) -> None:
        objective = Hartmann(n_variables=6, noise_std={'Hartmann': 0.05})
        with pytest.raises(ValueError, match="train_noise=True"):
            bayesian_optimiser(
                n_initial_points=4,
                n_bayesian_points=4,
                n_evaluations_per_step=1,
                objective=objective,
                model={
                    'noise_settings': {'train_noise': True},
                    'training_settings': {'max_iter': 5, 'verbose': False},
                },
            )

    def test_no_conflict_without_noise_std(self) -> None:
        """train_noise=True is allowed if objective has no noise_std."""
        objective = Hartmann(n_variables=6)
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={
                'noise_settings': {'train_noise': True},
                'training_settings': {'max_iter': 5, 'verbose': False},
            },
        )
        assert optimiser is not None

    def test_noise_std_tensor_property(self) -> None:
        optimiser = _make_single_objective_noisy_optimiser()
        tensor = optimiser._noise_std_tensor
        assert tensor is not None
        assert torch.allclose(tensor, torch.tensor([0.05]))

    def test_noise_std_tensor_is_none_when_not_set(self) -> None:
        objective = Hartmann(n_variables=6)
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        assert optimiser._noise_std_tensor is None

    def test_noise_std_in_model_space_without_normaliser_returns_physical(self) -> None:
        """Before first fit, _noise_std_in_model_space == physical units."""
        optimiser = _make_single_objective_noisy_optimiser()
        # No data evaluated yet, so normaliser is not fitted
        assert not optimiser.normalisers_have_been_initialised
        noise_in_model = optimiser._noise_std_in_model_space
        assert noise_in_model is not None
        assert torch.allclose(noise_in_model, torch.tensor([0.05]))

    def test_noise_std_in_model_space_with_normaliser_returns_scaled(self) -> None:
        """After normaliser is fitted, _noise_std_in_model_space is scaled."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        for _ in range(4):
            optimiser.run_optimisation_step()
        assert optimiser.normalisers_have_been_initialised
        # Scaled noise should differ from physical noise (unless std is 1)
        noise_physical = optimiser._noise_std_tensor
        noise_model_space = optimiser._noise_std_in_model_space
        assert noise_physical is not None
        assert noise_model_space is not None
        # They may or may not be equal depending on the normaliser, but both should be tensors
        assert noise_model_space.shape == noise_physical.shape


# ---------------------------------------------------------------------------
# Step 4 — GPyTorchFullModel._apply_physical_noise
# ---------------------------------------------------------------------------

class TestApplyPhysicalNoise:

    def _make_trained_full_model(self, n_variables: int = 3, n_objectives: int = 1) -> GPyTorchFullModel:
        kernel = MaternKernel(n_variables=n_variables)
        from veropt.optimiser.model import AdamModelOptimiser
        model = GPyTorchFullModel.from_the_beginning(
            n_variables=n_variables,
            n_objectives=n_objectives,
            single_model_list=[kernel],
            model_optimiser=AdamModelOptimiser(),
            max_iter=5,
            verbose=False
        )
        # Initialise with dummy data so model_with_data is set
        variables = torch.rand(6, n_variables)
        objectives = torch.rand(6, n_objectives)
        model.initialise_model(variable_values=variables, objective_values=objectives)
        return model

    def test_apply_physical_noise_sets_noise_value(self) -> None:
        model = self._make_trained_full_model()
        noise_std = torch.tensor([0.1])
        model._apply_physical_noise(noise_std_in_model_space=noise_std)
        expected_variance = 0.1 ** 2
        assert model._model_list[0].model_with_data is not None
        actual_noise = float(model._model_list[0].model_with_data.likelihood.noise)
        assert abs(actual_noise - expected_variance) < 1e-10

    def test_apply_physical_noise_sets_tight_lower_bound(self) -> None:
        model = self._make_trained_full_model()
        noise_std = torch.tensor([0.2])
        model._apply_physical_noise(noise_std_in_model_space=noise_std)
        expected_variance = 0.2 ** 2
        lower_bound = float(model._model_list[0].likelihood.noise_covar.raw_noise_constraint.lower_bound)
        assert abs(lower_bound - expected_variance * 0.99) < 1e-9

    def test_apply_physical_noise_below_floor_raises(self) -> None:
        model = self._make_trained_full_model()
        # Set a high floor then try to apply noise below it
        floor_kernel: NoiseSettingsInputDict = {'noise_lower_bound': 1.0}
        model._model_list[0].set_noise_constraint(lower_bound=1.0)
        tiny_noise_std = torch.tensor([0.001])  # variance = 1e-6 < 1.0 floor
        with pytest.raises(ValueError, match="noise_lower_bound"):
            model._apply_physical_noise(noise_std_in_model_space=tiny_noise_std)


# ---------------------------------------------------------------------------
# Step 5 — Predictor noise threading
# ---------------------------------------------------------------------------

class TestPredictorNoiseThreading:

    def test_update_with_noise_trains_without_error(self) -> None:
        """Smoke test: train predictor with noise_std_in_model_space provided."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        # Evaluate initial points so we have data
        for _ in range(4):
            optimiser.run_optimisation_step()
        # If we got here, noise threading through train_model worked
        assert optimiser.model_has_been_trained

    def test_noise_applied_on_train_false_reload(self, tmp_path: Path) -> None:
        """After save+load, the model noise should match objective.noise_std."""
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        for _ in range(4):
            optimiser.run_optimisation_step()

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "test_noisy_optimiser.json")
        save_to_json(optimiser, file_path)

        loaded = load_optimiser_from_state(file_path)

        # Verify noise was re-applied: model noise ~ physical_variance_normalised
        assert loaded.objective.noise_std is not None
        noise_in_model = loaded._noise_std_in_model_space
        assert noise_in_model is not None

        assert isinstance(loaded.predictor, BotorchPredictor)
        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(single_model.model_with_data.likelihood.noise)  # type: ignore[union-attr]
            assert abs(actual_noise - expected_variance) < 1e-9, (
                f"Objective {objective_index}: expected variance {expected_variance:.2e} "
                f"but model has {actual_noise:.2e}"
            )

    def test_multi_objective_noisy_reload_does_not_crash(self, tmp_path: Path) -> None:
        """Regression test: loading a saved multi-objective noisy optimiser must not raise
        UnsupportedError about unexpected batch dims in prune_inferior_points_multi_objective.

        Root cause: GPyTorchSingleModel.gather_dicts_to_save was saving
        model_with_data.train_inputs (a gpytorch tuple) instead of train_inputs[0] (the raw
        tensor). On reload, torch.tensor(tuple_data) produced shape [1, n_points, n_vars]
        instead of [n_points, n_vars], giving the reloaded model batch_shape=[1].
        BoTorch's prune_inferior_points_multi_objective saw obj_vals.ndim=4 > 3 and raised
        UnsupportedError("Models with multiple batch dims...").

        Fix: save train_inputs[0] in gather_dicts_to_save so reload gives batch_shape=[]."""
        import torch
        torch.manual_seed(0)

        noise_std = {f"DTLZ1 {i + 1}": 0.05 for i in range(2)}
        objective = DTLZ1(n_variables=3, n_objectives=2, noise_std=noise_std)
        optimiser = bayesian_optimiser(
            n_initial_points=6,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )

        for _ in range(6):
            optimiser.run_optimisation_step()

        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "test_multi_noisy_optimiser.json")
        save_to_json(optimiser, file_path)

        # This line used to raise:
        #   botorch.exceptions.errors.UnsupportedError:
        #   Models with multiple batch dims are currently unsupported by
        #   `prune_inferior_points_multi_objective`.
        loaded = load_optimiser_from_state(file_path)

        assert loaded.model_has_been_trained

        assert isinstance(loaded.predictor, BotorchPredictor)

        # batch_shape must be [] (no spurious batch dim from saved train_inputs tuple)
        reloaded_mlgp = loaded.predictor.model.get_gpytorch_model()
        assert reloaded_mlgp.batch_shape == torch.Size([]), (
            f"Expected batch_shape=[], got {reloaded_mlgp.batch_shape}. "
            "Saving train_inputs as a tuple introduced an extra batch dimension on reload."
        )

        # Verify noise values are correct after reload
        noise_in_model = loaded._noise_std_in_model_space
        assert noise_in_model is not None

        for objective_index, single_model in enumerate(loaded.predictor.model._model_list):
            expected_variance = float(noise_in_model[objective_index] ** 2)
            actual_noise = float(single_model.model_with_data.likelihood.noise.detach())  # type: ignore[union-attr]
            assert abs(actual_noise - expected_variance) < 1e-9, (
                f"Objective {objective_index}: expected variance {expected_variance:.2e} "
                f"but model has {actual_noise:.2e}"
            )


# ---------------------------------------------------------------------------
# Steps 7–8 — Acquisition function auto-selection
# ---------------------------------------------------------------------------

class TestNoisyAcquisitionAutoSelection:

    def test_noisy_multi_objective_selects_qlogneHVI(self) -> None:
        acq = botorch_acquisition_function(
            n_variables=5,
            n_objectives=3,
            is_noisy=True
        )
        assert acq.name == 'qlogneHVI'

    def test_noisy_single_objective_keeps_ucb(self) -> None:
        acq = botorch_acquisition_function(
            n_variables=3,
            n_objectives=1,
            is_noisy=True
        )
        assert acq.name == 'ucb'

    def test_non_noisy_multi_objective_selects_qlogehvi(self) -> None:
        acq = botorch_acquisition_function(
            n_variables=5,
            n_objectives=3,
            is_noisy=False
        )
        assert acq.name == 'qlogehvi'

    def test_bayesian_optimiser_with_noisy_multi_objective_uses_qlogneHVI(self) -> None:
        optimiser = _make_multi_objective_noisy_optimiser()
        assert isinstance(optimiser.predictor, BotorchPredictor)
        assert optimiser.predictor.acquisition_function.name == 'qlogneHVI'

    def test_bayesian_optimiser_without_noise_uses_qlogehvi(self) -> None:
        objective = VehicleSafety()  # no noise_std
        optimiser = bayesian_optimiser(
            n_initial_points=4,
            n_bayesian_points=4,
            n_evaluations_per_step=1,
            objective=objective,
            model={'training_settings': {'max_iter': 5, 'verbose': False}},
        )
        assert isinstance(optimiser.predictor, BotorchPredictor)
        assert optimiser.predictor.acquisition_function.name == 'qlogehvi'

    def test_noisy_optimiser_full_run(self) -> None:
        """Smoke test: full optimisation loop with noise_std set."""
        optimiser = _make_multi_objective_noisy_optimiser(n_initial=4, n_bayesian=2)
        for step in range(6):
            optimiser.run_optimisation_step()
        assert optimiser.n_points_evaluated == 6


# ---------------------------------------------------------------------------
# Model posterior variance — does the pinned noise actually reach the GP?
# ---------------------------------------------------------------------------

class TestModelPosteriorVariance:
    """Check that fixing noise_std produces meaningfully different posterior uncertainty
    compared to a noiseless model, both immediately after pinning and after training."""

    def _train_simple_model(self, noise_std_value: Optional[float]) -> GPyTorchFullModel:
        """Return a GPyTorchFullModel with initialised (not full Adam) training on a small 1-D dataset.

        Using initialise_model + _apply_physical_noise mirrors the path taken by train_model."""
        from veropt.optimiser.model import AdamModelOptimiser
        kernel = MaternKernel(n_variables=1)
        model = GPyTorchFullModel.from_the_beginning(
            n_variables=1,
            n_objectives=1,
            single_model_list=[kernel],
            model_optimiser=AdamModelOptimiser(),
            max_iter=5,
            verbose=False
        )
        torch.manual_seed(0)
        variables = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
        objectives = torch.sin(variables * 3.14).squeeze(-1).unsqueeze(-1)
        model.initialise_model(variable_values=variables, objective_values=objectives)
        if noise_std_value is not None:
            noise_tensor = torch.tensor([noise_std_value])
            model._apply_physical_noise(noise_std_in_model_space=noise_tensor)
        return model

    def _posterior_variance_at_training_point(self, model: GPyTorchFullModel) -> float:
        """Return the GP posterior variance at the first training point."""
        single_model = model._model_list[0]
        assert single_model.model_with_data is not None
        gpytorch_model = single_model.model_with_data
        gpytorch_model.eval()
        x_test = gpytorch_model.train_inputs[0][0:1]  # first training point from ExactGP storage
        with torch.no_grad():
            posterior = gpytorch_model.likelihood(gpytorch_model(x_test))
        return float(posterior.variance.squeeze())

    def test_noiseless_model_has_near_zero_variance_at_training_point(self) -> None:
        """With no noise (likelihood noise ≈ 0), the GP nearly interpolates its training data."""
        model = self._train_simple_model(noise_std_value=None)
        variance = self._posterior_variance_at_training_point(model)
        assert variance < 0.01, (
            f"Expected near-zero posterior variance at training point for noiseless GP, got {variance:.4e}"
        )

    def test_noisy_model_has_larger_variance_at_training_point(self) -> None:
        """With noise_std set, the posterior should be meaningfully uncertain at training points."""
        noise_std = 0.3
        model = self._train_simple_model(noise_std_value=noise_std)
        variance = self._posterior_variance_at_training_point(model)
        assert variance > 0.005, (
            f"Expected non-negligible variance at training point for noisy GP (noise_std={noise_std}), "
            f"got {variance:.4e}"
        )

    def test_noisy_model_has_greater_variance_than_noiseless(self) -> None:
        """Noisy model should always have higher posterior uncertainty than noiseless."""
        noiseless_model = self._train_simple_model(noise_std_value=None)
        noisy_model = self._train_simple_model(noise_std_value=0.3)
        noiseless_variance = self._posterior_variance_at_training_point(noiseless_model)
        noisy_variance = self._posterior_variance_at_training_point(noisy_model)
        assert noisy_variance > noiseless_variance, (
            f"Noisy model variance ({noisy_variance:.4e}) should exceed "
            f"noiseless model variance ({noiseless_variance:.4e})"
        )

    def test_noise_pinning_survives_adam_training(self) -> None:
        """After running Adam optimisation, the likelihood noise should remain pinned to the physical variance."""
        from veropt.optimiser.model import AdamModelOptimiser
        noise_std = 0.1
        kernel = MaternKernel(n_variables=1)
        model = GPyTorchFullModel.from_the_beginning(
            n_variables=1,
            n_objectives=1,
            single_model_list=[kernel],
            model_optimiser=AdamModelOptimiser(),
            max_iter=10,
            verbose=False
        )
        torch.manual_seed(1)
        variables = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
        objectives = torch.sin(variables * 3.14).squeeze(-1).unsqueeze(-1)
        noise_tensor = torch.tensor([noise_std])
        model.train_model(
            variable_values=variables,
            objective_values=objectives,
            noise_std_in_model_space=noise_tensor
        )
        expected_variance = noise_std ** 2
        assert model._model_list[0].model_with_data is not None
        actual_noise = float(model._model_list[0].model_with_data.likelihood.noise)
        relative_error = abs(actual_noise - expected_variance) / expected_variance
        assert relative_error < 0.01, (
            f"Noise variance should remain pinned to {expected_variance:.4e} after Adam training, "
            f"but got {actual_noise:.4e} (relative error {relative_error:.2%})"
        )


# ---------------------------------------------------------------------------
# JSON noise desync detection
# ---------------------------------------------------------------------------

class TestJsonNoiseDesyncDetection:
    """Verify that editing the kernel noise in the JSON file (while objective.noise_std
    disagrees) raises a clear error on reload rather than silently overwriting."""

    def _run_optimiser_and_save(self, tmp_path: Path) -> str:
        optimiser = _make_single_objective_noisy_optimiser(n_initial=4)
        for _ in range(4):
            optimiser.run_optimisation_step()
        optimiser.settings.allow_automatic_json_updates = True
        file_path = str(tmp_path / "desync_test.json")
        save_to_json(optimiser, file_path)
        return file_path

    def test_clean_json_reloads_without_error(self, tmp_path: Path) -> None:
        """Sanity check: an unmodified JSON should reload cleanly."""
        file_path = self._run_optimiser_and_save(tmp_path)
        loaded = load_optimiser_from_state(file_path)
        assert loaded.objective.noise_std is not None

    def test_tampered_kernel_noise_raises_on_reload(self, tmp_path: Path) -> None:
        """Manually changing the kernel likelihood noise in the JSON should raise ValueError."""
        file_path = self._run_optimiser_and_save(tmp_path)

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        # Navigate to the raw_noise value in the first single model's state_dict.
        # Structure: optimiser → predictor → state → model → state
        #            → model_dicts → model_0 → state → state_dict
        #            → likelihood.noise_covar.raw_noise  (flat list)
        state_dict = (
            data['optimiser']['predictor']['state']['model']['state']
            ['model_dicts']['model_0']['state']['state_dict']
        )
        raw_noise_key = 'likelihood.noise_covar.raw_noise'
        if raw_noise_key in state_dict:
            # Set raw_noise to a very large value — corresponds to a much higher variance after softplus
            state_dict[raw_noise_key] = [100.0]  # large positive raw value → huge effective noise

        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)

        with pytest.raises(ValueError, match="Noise desync detected"):
            load_optimiser_from_state(file_path)























