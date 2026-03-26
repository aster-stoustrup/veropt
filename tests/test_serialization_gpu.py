"""Tests for GPU tensor serialization and device handling.

Ensures that all saved state is CPU-only and can be properly restored to the current device.
"""

import torch

from veropt.optimiser.device_manager import (
    initialize, DeviceConfig, reset, 
    tensors_to_cpu, tensors_to_device
)
from veropt.optimiser.optimiser_utility import OptimiserSettings, SuggestedPoints


# Tests for tensors_to_cpu and tensors_to_device helper functions


def test_tensors_to_cpu_single_tensor() -> None:
    """Test converting a single tensor to CPU."""
    tensor = torch.randn(3, 4)
    result = tensors_to_cpu(tensor)
    assert result.device.type == 'cpu'
    assert torch.allclose(result, tensor.cpu())


def test_tensors_to_cpu_dict() -> None:
    """Test converting tensors in a dict to CPU."""
    data = {
        'a': torch.randn(2),
        'b': torch.randn(3, 4),
        'c': 'not a tensor',
        'd': 42,
    }
    result = tensors_to_cpu(data)
    assert isinstance(result, dict)
    assert result['a'].device.type == 'cpu'
    assert result['b'].device.type == 'cpu'
    assert result['c'] == 'not a tensor'
    assert result['d'] == 42


def test_tensors_to_cpu_nested_structures() -> None:
    """Test converting tensors in nested lists and dicts."""
    data = {
        'tensors': [torch.randn(2), torch.randn(3)],
        'nested': {
            'tensor': torch.randn(4),
            'list': [torch.randn(5), 'string']
        }
    }
    result = tensors_to_cpu(data)
    assert result['tensors'][0].device.type == 'cpu'
    assert result['tensors'][1].device.type == 'cpu'
    assert result['nested']['tensor'].device.type == 'cpu'
    assert result['nested']['list'][0].device.type == 'cpu'
    assert result['nested']['list'][1] == 'string'


def test_tensors_to_cpu_idempotent() -> None:
    """Test that tensors_to_cpu is idempotent (can call multiple times)."""
    tensor = torch.randn(3, 4)
    result1 = tensors_to_cpu(tensor)
    result2 = tensors_to_cpu(result1)
    assert torch.allclose(result1, result2)
    assert result2.device.type == 'cpu'


def test_tensors_to_device_single_tensor() -> None:
    """Test converting a single tensor to device."""
    tensor = torch.randn(3, 4)
    device = torch.device('cpu')
    result = tensors_to_device(tensor, device=device)
    assert result.device == device


def test_tensors_to_device_dict() -> None:
    """Test converting tensors in a dict to device."""
    data = {
        'a': torch.randn(2),
        'b': torch.randn(3, 4),
        'c': 'not a tensor',
    }
    device = torch.device('cpu')
    result = tensors_to_device(data, device=device)
    assert result['a'].device == device
    assert result['b'].device == device
    assert result['c'] == 'not a tensor'


def test_tensors_to_device_idempotent() -> None:
    """Test that tensors_to_device is idempotent."""
    tensor = torch.randn(3, 4)
    device = torch.device('cpu')
    result1 = tensors_to_device(tensor, device=device)
    result2 = tensors_to_device(result1, device=device)
    assert torch.allclose(result1, result2)
    assert result2.device == device



# Tests for OptimiserSettings serialization with GPU support


def test_objective_weights_stays_cpu_on_save() -> None:
    """Test that objective_weights is CPU when saved."""
    settings = OptimiserSettings(
        n_initial_points=10,
        n_bayesian_points=50,
        n_objectives=2,
        n_evaluations_per_step=1,
        objective_weights=[0.6, 0.4]
    )
    saved = settings.gather_dicts_to_save()
    assert saved['objective_weights'].device.type == 'cpu'


def test_settings_roundtrip() -> None:
    """Test that settings can be saved and loaded correctly."""
    original_settings = OptimiserSettings(
        n_initial_points=15,
        n_bayesian_points=75,
        n_objectives=3,
        n_evaluations_per_step=1,
        objective_weights=[0.5, 0.3, 0.2]
    )
    saved = original_settings.gather_dicts_to_save()
    restored = OptimiserSettings.from_saved_state(saved)
    
    assert restored.n_initial_points == original_settings.n_initial_points
    assert restored.n_bayesian_points == original_settings.n_bayesian_points
    assert torch.allclose(restored.objective_weights, original_settings.objective_weights)


# Tests for SuggestedPoints serialization with GPU support


def test_copy_creates_cpu_tensors() -> None:
    """Test that copy() creates CPU tensors."""
    points = SuggestedPoints(
        variable_values=torch.randn(5, 3),
        predicted_objective_values={
            'mean': torch.randn(5),
            'lower': torch.randn(5),
            'upper': torch.randn(5),
        },
        generated_at_step=1,
        generated_with_mode='bayesian',
        normalised=False
    )
    copied = points.copy()
    
    assert copied.variable_values.device.type == 'cpu'
    assert copied.predicted_objective_values['mean'].device.type == 'cpu'
    assert copied.predicted_objective_values['lower'].device.type == 'cpu'
    assert copied.predicted_objective_values['upper'].device.type == 'cpu'


def test_copy_preserves_values() -> None:
    """Test that copy() preserves tensor values."""
    original_values = torch.randn(5, 3)
    original_preds = {
        'mean': torch.randn(5),
        'lower': torch.randn(5),
        'upper': torch.randn(5),
    }
    points = SuggestedPoints(
        variable_values=original_values,
        predicted_objective_values=original_preds,
        generated_at_step=1,
        generated_with_mode='bayesian',
        normalised=False
    )
    copied = points.copy()
    
    assert torch.allclose(copied.variable_values, original_values)
    assert torch.allclose(copied.predicted_objective_values['mean'], original_preds['mean'])
    assert torch.allclose(copied.predicted_objective_values['lower'], original_preds['lower'])
    assert torch.allclose(copied.predicted_objective_values['upper'], original_preds['upper'])


def test_copy_with_none_predictions() -> None:
    """Test copy() when predicted_objective_values is None."""
    points = SuggestedPoints(
        variable_values=torch.randn(5, 3),
        predicted_objective_values=None,
        generated_at_step=0,
        generated_with_mode='initial',
        normalised=False
    )
    copied = points.copy()
    
    assert copied.predicted_objective_values is None
    assert copied.variable_values.device.type == 'cpu'


# Integration tests for device manager with serialization


def test_cpu_default_preserves_cpu_tensors() -> None:
    """Test that CPU-only mode preserves CPU tensors in saved state."""
    reset()
    initialize(DeviceConfig(use_gpu=False))
    
    try:
        settings = OptimiserSettings(
            n_initial_points=10,
            n_bayesian_points=50,
            n_objectives=2,
            n_evaluations_per_step=1
        )
        saved = settings.gather_dicts_to_save()
        assert saved['objective_weights'].device.type == 'cpu'
    finally:
        reset()


def test_serialization_roundtrip_preserves_dtype() -> None:
    """Test that dtype is preserved during serialization."""
    # Create tensors with float64 (default dtype)
    tensor_dict = {
        'weights': torch.ones(5, dtype=torch.float64),
        'values': torch.randn(3, 4, dtype=torch.float64)
    }
    
    cpu_dict = tensors_to_cpu(tensor_dict)
    
    assert cpu_dict['weights'].dtype == torch.float64
    assert cpu_dict['values'].dtype == torch.float64
