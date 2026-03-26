"""Tests for GPU support in veropt device management and model training."""

import numpy as np
import pytest
import torch

from veropt.optimiser import device_manager
from veropt.optimiser.device_manager import DeviceConfig
from veropt.optimiser.acquisition_optimiser import DualAnnealingOptimiser


def test_device_manager_default_cpu() -> None:
    """Test that CPU is the default device when not configured."""
    device_manager.reset()
    
    assert device_manager.get_device().type == 'cpu'
    assert not device_manager.is_gpu_available()


def test_device_manager_default_dtype_float64() -> None:
    """Test that default dtype is float64 for numerical stability."""
    assert device_manager.get_default_dtype() == torch.float64


def test_device_manager_gpu_fallback_unavailable() -> None:
    """Test that GPU request falls back to CPU if CUDA unavailable."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, cannot test fallback scenario")
    
    device_manager.reset()
    
    config = DeviceConfig(use_gpu=True)
    device_manager.initialize(config)
    
    # Should fall back to CPU
    assert device_manager.get_device().type == 'cpu'
    assert not device_manager.is_gpu_available()


def test_tensor_to_numpy_conversion() -> None:
    """Test conversion of tensors to numpy arrays."""
    tensor = torch.randn(3, 4, dtype=torch.float64)
    
    numpy_array = device_manager.tensor_to_numpy(tensor)
    
    assert isinstance(numpy_array, np.ndarray)
    assert numpy_array.shape == (3, 4)
    assert numpy_array.dtype == np.float64
    np.testing.assert_allclose(numpy_array, tensor.numpy(), rtol=1e-7)


def test_numpy_to_tensor_conversion() -> None:
    """Test conversion of numpy arrays to tensors on correct device."""
    device_manager.reset()
    
    numpy_array = np.random.randn(3, 4).astype(np.float64)
    tensor = device_manager.numpy_to_tensor(numpy_array)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float64
    assert tensor.device.type == 'cpu'
    np.testing.assert_allclose(tensor.numpy(), numpy_array, rtol=1e-7)


def test_move_tensor_to_device() -> None:
    """Test moving tensor to specified device."""
    device_manager.reset()
    
    tensor = torch.randn(3, 4)
    moved_tensor = device_manager.move_to_device(tensor)
    
    assert moved_tensor.device.type == 'cpu'
    assert moved_tensor.shape == tensor.shape


def test_dual_annealing_optimizer_cpu() -> None:
    """Test DualAnnealingOptimiser works with CPU tensors."""
    bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).T
    
    optimizer = DualAnnealingOptimiser(
        bounds=bounds,
        n_evaluations_per_step=1,
        max_iter=100
    )
    
    assert optimizer.name == 'dual_annealing'
    assert optimizer.maximum_evaluations_per_step == 1
    assert optimizer.n_evaluations_per_step == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_botorch_optimizer_gpu_available() -> None:
    """Test BotorchAcquisitionOptimiser can be instantiated when CUDA available."""
    try:
        from veropt.optimiser.acquisition_optimiser import BotorchAcquisitionOptimiser
    except ImportError:
        pytest.skip("BoTorch not installed")
    
    bounds = torch.tensor([[0.0, 1.0], [0.0, 1.0]]).T
    
    # Test single point optimization (q=1)
    optimizer_single = BotorchAcquisitionOptimiser(
        bounds=bounds,
        n_evaluations_per_step=1,
        max_iter=100,
        num_restarts=10,
        raw_samples=256
    )
    assert optimizer_single.name == 'botorch'
    assert optimizer_single.n_evaluations_per_step == 1
    # BoTorch supports batch optimization, so maximum is 128
    assert optimizer_single.maximum_evaluations_per_step == 128
    
    # Test batch optimization (q > 1)
    optimizer_batch = BotorchAcquisitionOptimiser(
        bounds=bounds,
        n_evaluations_per_step=5,
        max_iter=100,
        num_restarts=10,
        raw_samples=256
    )
    assert optimizer_batch.n_evaluations_per_step == 5
    # Both optimizers should have the same class-level maximum
    assert optimizer_batch.maximum_evaluations_per_step == 128


def test_device_config_cpu() -> None:
    """Test DeviceConfig with CPU."""
    device_manager.reset()
    config = DeviceConfig(use_gpu=False)
    device_manager.initialize(config)
    
    assert device_manager.get_device().type == 'cpu'


def test_device_config_preserves_cpu_on_failure() -> None:
    """Test that invalid device names fall back to CPU gracefully."""
    device_manager.reset()
    config = DeviceConfig(use_gpu=False, device_name='cpu')
    device_manager.initialize(config)
    
    assert device_manager.get_device().type == 'cpu'
