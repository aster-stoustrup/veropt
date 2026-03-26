"""Tests for acquisition optimiser implementations."""

import numpy as np
import pytest
import torch

from veropt.optimiser.acquisition import AcquisitionFunction
from veropt.optimiser.acquisition_optimiser import (
    DualAnnealingOptimiser,
    BotorchAcquisitionOptimiser,
    ProximityPunishAcquisitionFunction,
    ProximityPunishmentSequentialOptimiser,
    TorchNumpyWrapper,
    _BotorchAcqFuncWrapper,
)
from veropt.optimiser.saver_loader_utility import SavableDataClass


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_bounds() -> torch.Tensor:
    """Simple 2D bounds for testing."""
    return torch.tensor([[0.0, 1.0], [0.0, 1.0]]).T


@pytest.fixture
def simple_acquisition_function() -> AcquisitionFunction:
    """Simple quadratic acquisition function (peak at center)."""
    class SimpleAcqFunc(AcquisitionFunction):
        name = 'simple_quad'
        multi_objective = False
        
        def __call__(self, variable_values: torch.Tensor) -> torch.Tensor:
            # Peak at [0.5, 0.5], quadratic decay
            center = torch.tensor([0.5, 0.5])
            distances = torch.norm(variable_values - center, dim=1)
            return 1.0 - distances ** 2
        
        def get_settings(self) -> SavableDataClass:
            """Return empty settings for simple test function."""
            from veropt.optimiser.saver_loader_utility import EmptyDataClass
            return EmptyDataClass()
        
        @classmethod
        def from_n_variables_n_objectives_and_settings(
                cls,
                n_variables: int,
                n_objectives: int,
                settings: dict
        ) -> 'SimpleAcqFunc':
            """Construct from parameters (settings ignored for test)."""
            return cls()
    
    return SimpleAcqFunc(n_variables=2, n_objectives=1)


@pytest.fixture
def training_inputs() -> torch.Tensor:
    """Simple training data for proximity punishment."""
    return torch.tensor([
        [0.2, 0.2],
        [0.8, 0.8],
    ], dtype=torch.float64)


# ============================================================================
# TorchNumpyWrapper Tests
# ============================================================================

def test_torch_numpy_wrapper_1d_input(simple_acquisition_function) -> None:
    """Test wrapper handles 1D input correctly."""
    wrapper = TorchNumpyWrapper(simple_acquisition_function)
    
    # Single point as 1D array
    x = np.array([0.5, 0.5])
    result = wrapper(x)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)  # Wrapped to [1, 2] internally, output [1]
    assert result[0] > 0.9  # Should be high near center


def test_torch_numpy_wrapper_2d_input(simple_acquisition_function) -> None:
    """Test wrapper handles 2D input correctly."""
    wrapper = TorchNumpyWrapper(simple_acquisition_function)
    
    # Multiple points
    x = np.array([[0.5, 0.5], [0.0, 0.0]])
    result = wrapper(x)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] > result[1]  # Center should score higher


def test_torch_numpy_wrapper_dtype_preserved(simple_acquisition_function) -> None:
    """Test wrapper preserves float64 dtype."""
    wrapper = TorchNumpyWrapper(simple_acquisition_function)
    
    x = np.array([[0.5, 0.5]], dtype=np.float64)
    result = wrapper(x)
    
    assert result.dtype == np.float64


# ============================================================================
# DualAnnealingOptimiser Tests
# ============================================================================

def test_dual_annealing_optimiser_basic(simple_bounds, simple_acquisition_function) -> None:
    """Test DualAnnealingOptimiser finds reasonable optimum."""
    optimiser = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=500
    )
    
    candidates = optimiser.optimise(simple_acquisition_function)
    
    # Should return single point
    assert candidates.shape == (1, 2)
    
    # Should be near center [0.5, 0.5]
    center = torch.tensor([0.5, 0.5])
    distance = torch.norm(candidates[0] - center)
    assert distance < 0.2, f"Optimizer found {candidates[0]}, far from center"


def test_dual_annealing_optimiser_call_method(simple_bounds, simple_acquisition_function) -> None:
    """Test __call__ method delegates to optimise."""
    optimiser = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=100
    )
    
    # Both should work
    result1 = optimiser.optimise(simple_acquisition_function)
    result2 = optimiser(simple_acquisition_function)
    
    # Shapes should match (values may differ due to randomness)
    assert result1.shape == result2.shape


def test_dual_annealing_optimiser_respects_bounds(simple_acquisition_function) -> None:
    """Test optimizer respects specified bounds."""
    bounded = torch.tensor([[0.3, 0.7], [0.2, 0.8]]).T
    optimiser = DualAnnealingOptimiser(
        bounds=bounded,
        n_evaluations_per_step=1,
        max_iter=500
    )
    
    candidates = optimiser.optimise(simple_acquisition_function)
    
    # Check bounds are respected
    assert (candidates[0, 0] >= 0.3) and (candidates[0, 0] <= 0.7)
    assert (candidates[0, 1] >= 0.2) and (candidates[0, 1] <= 0.8)


def test_dual_annealing_get_settings(simple_bounds) -> None:
    """Test get_settings returns correct settings."""
    optimiser = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=2000
    )
    
    settings = optimiser.get_settings()
    assert settings.max_iter == 2000


def test_dual_annealing_from_saved_state(simple_bounds) -> None:
    """Test creating optimizer from saved state."""
    original = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=1500
    )
    
    saved = original.gather_dicts_to_save()
    restored = DualAnnealingOptimiser.from_saved_state(saved['state'])
    
    assert restored.name == original.name
    assert torch.allclose(restored.bounds, original.bounds)
    assert restored.n_evaluations_per_step == original.n_evaluations_per_step


def test_dual_annealing_rejects_invalid_n_evaluations(simple_bounds) -> None:
    """Test optimizer rejects n_evaluations_per_step > maximum."""
    with pytest.raises(AssertionError):
        DualAnnealingOptimiser(
            bounds=simple_bounds,
            n_evaluations_per_step=5,  # Exceeds maximum_evaluations_per_step=1
            max_iter=1000
        )


# ============================================================================
# BotorchAcquisitionOptimiser Tests
# ============================================================================

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available (BotorchAcquisitionOptimiser designed for GPU)"
)
def test_botorch_optimiser_available_has_botorch(simple_bounds) -> None:
    """Test BotorchAcquisitionOptimiser requires BoTorch."""
    try:
        import botorch  # noqa
    except ImportError:
        pytest.skip("BoTorch not installed")


def test_botorch_optimiser_basic(simple_bounds, simple_acquisition_function) -> None:
    """Test BotorchAcquisitionOptimiser finds reasonable optimum."""
    try:
        import botorch  # noqa
    except ImportError:
        pytest.skip("BoTorch not installed")
    
    optimiser = BotorchAcquisitionOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=50,
        num_restarts=5,
        raw_samples=64
    )
    
    candidates = optimiser.optimise(simple_acquisition_function)
    
    # Should return single point
    assert candidates.shape == (1, 2)
    
    # Should be near center [0.5, 0.5]
    center = torch.tensor([0.5, 0.5])
    distance = torch.norm(candidates[0] - center)
    assert distance < 0.3


def test_botorch_optimiser_batch_mode(simple_bounds, simple_acquisition_function) -> None:
    """Test BotorchAcquisitionOptimiser supports batch optimization."""
    try:
        import botorch  # noqa
    except ImportError:
        pytest.skip("BoTorch not installed")
    
    optimiser = BotorchAcquisitionOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=3,  # Batch of 3
        max_iter=30,
        num_restarts=3,
        raw_samples=32
    )
    
    candidates = optimiser.optimise(simple_acquisition_function)
    
    # Should return 3 points
    assert candidates.shape == (3, 2)
    
    # All should respect bounds
    assert torch.all(candidates >= simple_bounds[0])
    assert torch.all(candidates <= simple_bounds[1])


def test_botorch_optimiser_maximum_evaluations(simple_bounds) -> None:
    """Test maximum_evaluations_per_step is set correctly."""
    try:
        import botorch  # noqa
    except ImportError:
        pytest.skip("BoTorch not installed")
    
    optimiser = BotorchAcquisitionOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=5,
        max_iter=50
    )
    
    assert optimiser.maximum_evaluations_per_step == 5


# ============================================================================
# _BotorchAcqFuncWrapper Tests
# ============================================================================

def test_botorch_wrapper_1d_conversion(simple_acquisition_function) -> None:
    """Test wrapper correctly converts tensor formats."""
    wrapper = _BotorchAcqFuncWrapper(simple_acquisition_function, q=1)
    
    # BoTorch format: [batch_size=2, q=1, n_vars=2]
    x_botorch = torch.tensor([[[0.5, 0.5]], [[0.0, 0.0]]], dtype=torch.float64)
    
    result = wrapper(x_botorch)
    
    # Output should be [batch_size=2]
    assert result.shape == (2,)
    
    # Values should be positive (acquisition function is non-negative)
    assert torch.all(result >= 0)


def test_botorch_wrapper_batch_aggregation(simple_acquisition_function) -> None:
    """Test wrapper aggregates batch correctly."""
    wrapper = _BotorchAcqFuncWrapper(simple_acquisition_function, q=2)
    
    # BoTorch format: [batch_size=1, q=2, n_vars=2]
    x_botorch = torch.tensor([[[0.5, 0.5], [0.0, 0.0]]], dtype=torch.float64)
    
    result = wrapper(x_botorch)
    
    # Output should be [batch_size=1]
    assert result.shape == (1,)
    assert result.item() > 0


# ============================================================================
# ProximityPunishAcquisitionFunction Tests
# ============================================================================

def test_proximity_punish_reduces_near_existing(simple_acquisition_function, training_inputs) -> None:
    """Test proximity punishment reduces values near existing points."""
    punished = ProximityPunishAcquisitionFunction(
        acquisition_function=simple_acquisition_function,
        mean_of_train_inputs=training_inputs
    )
    
    # Evaluate at existing point
    at_existing = punished(training_inputs[:1])
    
    # Evaluate far from existing points
    far_away = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    at_far = punished(far_away)
    
    # Far away should have higher acquisition value
    assert at_far[0] > at_existing[0]


def test_proximity_punish_calculation(simple_acquisition_function, training_inputs) -> None:
    """Test proximity punishment scaling is calculated."""
    punished = ProximityPunishAcquisitionFunction(
        acquisition_function=simple_acquisition_function,
        mean_of_train_inputs=training_inputs
    )
    
    # This should work without errors
    punished.calculate_scaling([training_inputs])
    
    assert punished.scaling is not None
    assert punished.scaling > 0


# ============================================================================
# ProximityPunishmentSequentialOptimiser Tests
# ============================================================================

def test_proximity_punish_optimiser_basic(simple_bounds, simple_acquisition_function, training_inputs) -> None:
    """Test ProximityPunishmentSequentialOptimiser avoids existing points."""
    base_optimiser = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=500
    )
    
    punish_optimiser = ProximityPunishmentSequentialOptimiser(
        acquisition_optimiser=base_optimiser,
        proximity_punishment_settings={},
        problematic_points=training_inputs
    )
    
    candidate = punish_optimiser.optimise(simple_acquisition_function)
    
    assert candidate.shape == (1, 2)
    
    # Should be far from training points
    distances_to_training = torch.cdist(candidate, training_inputs)
    min_distance = distances_to_training.min().item()
    assert min_distance > 0.1


def test_proximity_punish_optimiser_update_problematic_points(
    simple_bounds, simple_acquisition_function, training_inputs
) -> None:
    """Test updating problematic points."""
    base_optimiser = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=100
    )
    
    punish_optimiser = ProximityPunishmentSequentialOptimiser(
        acquisition_optimiser=base_optimiser,
        proximity_punishment_settings={},
        problematic_points=training_inputs[:1]
    )
    
    # Update with more points
    new_points = torch.cat([training_inputs[:1], training_inputs[1:]])
    punish_optimiser.update_problematic_points(new_points)
    
    assert torch.allclose(punish_optimiser.problematic_points, new_points)


# ============================================================================
# Integration Tests
# ============================================================================

def test_dual_annealing_with_proximity_punishment(simple_bounds, simple_acquisition_function) -> None:
    """Integration test: DualAnnealing + ProximityPunishment."""
    training_inputs = torch.tensor([[0.25, 0.25], [0.75, 0.75]]).T
    
    base_optimiser = DualAnnealingOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=1,
        max_iter=500
    )
    
    punish_optimiser = ProximityPunishmentSequentialOptimiser(
        acquisition_optimiser=base_optimiser,
        proximity_punishment_settings={},
        problematic_points=training_inputs
    )
    
    candidates = punish_optimiser.optimise(simple_acquisition_function)
    
    # Should return new point
    assert candidates.shape == (1, 2)
    
    # Should be different from training points
    distances = torch.cdist(candidates, training_inputs)
    assert distances.min().item() > 0.05


def test_botorch_batch_with_proximity_punishment(simple_bounds, simple_acquisition_function) -> None:
    """Integration test: BotorchOptimiser batch + ProximityPunishment."""
    try:
        import botorch  # noqa
    except ImportError:
        pytest.skip("BoTorch not installed")
    
    training_inputs = torch.tensor([[0.2, 0.2], [0.8, 0.8]]).T
    
    base_optimiser = BotorchAcquisitionOptimiser(
        bounds=simple_bounds,
        n_evaluations_per_step=2,  # Batch of 2
        max_iter=30,
        num_restarts=3,
        raw_samples=32
    )
    
    punish_optimiser = ProximityPunishmentSequentialOptimiser(
        acquisition_optimiser=base_optimiser,
        proximity_punishment_settings={},
        problematic_points=training_inputs
    )
    
    candidates = punish_optimiser.optimise(simple_acquisition_function)
    
    # Should return 2 points (batch)
    assert candidates.shape == (2, 2)
    
    # Both should respect bounds
    assert torch.all(candidates >= simple_bounds[0])
    assert torch.all(candidates <= simple_bounds[1])
