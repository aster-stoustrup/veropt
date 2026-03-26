"""
Device management for GPU/CPU tensor operations.

Provides centralized device and dtype management for PyTorch, GPyTorch, and BoTorch models.
Enables seamless GPU support while maintaining backwards compatibility with CPU-only execution.

Uses module-level singleton semantics (standard Python module caching).
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

# Module-level state (singleton via Python module caching)
_device: torch.device = torch.device('cpu')
_use_gpu: bool = False
_default_dtype: torch.dtype = torch.float64

# Initialize default dtype globally
torch.set_default_dtype(_default_dtype)


@dataclass
class DeviceConfig:
    """Configuration for device selection and dtype management.
    
    Attributes:
        use_gpu: If True, attempt to use CUDA. Falls back to CPU if unavailable.
        device_name: Optional CUDA device name (e.g., 'cuda:0'). If None, auto-selects.
    """
    use_gpu: bool = False
    device_name: Optional[str] = None


def initialize(config: DeviceConfig) -> None:
    """Initialize device manager with configuration.
    
    Args:
        config: DeviceConfig specifying GPU preference and device name.
    """
    global _device, _use_gpu
    
    if config.use_gpu:
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available. Falling back to CPU. "
                "Ensure NVIDIA driver and PyTorch CUDA build are installed.",
                RuntimeWarning,
                stacklevel=2
            )
            _device = torch.device('cpu')
            _use_gpu = False
        else:
            try:
                # Try to use specified device or auto-select cuda:0
                device_name = config.device_name or 'cuda:0'
                _device = torch.device(device_name)
                # Verify device is valid
                _ = torch.empty(0, device=_device)
                _use_gpu = True
            except RuntimeError as e:
                warnings.warn(
                    f"Failed to initialize device '{config.device_name or 'cuda:0'}': {e}. "
                    "Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2
                )
                _device = torch.device('cpu')
                _use_gpu = False
    else:
        _device = torch.device('cpu')
        _use_gpu = False


def get_device() -> torch.device:
    """Get the current device (cuda or cpu).
    
    Returns:
        torch.device instance.
    """
    return _device


def is_gpu_available() -> bool:
    """Check if GPU is currently enabled and available.
    
    Returns:
        True if GPU is active, False if on CPU.
    """
    return _use_gpu


def get_default_dtype() -> torch.dtype:
    """Get the default dtype for tensor operations.
    
    Returns:
        torch.dtype, always float64 for numerical stability.
    """
    return _default_dtype


def move_to_device(
    tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Move tensor to device with optional dtype conversion.
    
    Args:
        tensor: Tensor to move.
        device: Target device. If None, uses current device.
        dtype: Target dtype. If None, preserves original dtype.
        
    Returns:
        Tensor on target device with specified dtype.
    """
    target_device = device or _device
    target_dtype = dtype or tensor.dtype
    
    return tensor.to(device=target_device, dtype=target_dtype)


def numpy_to_tensor(
    array,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Convert numpy array to tensor on current device.
    
    Args:
        array: numpy array or array-like.
        dtype: Target dtype. If None, uses default dtype.
        
    Returns:
        torch.Tensor on current device.
    """
    target_dtype = dtype or get_default_dtype()
    tensor = torch.as_tensor(array, dtype=target_dtype)
    return tensor.to(device=_device)


def tensor_to_numpy(tensor: torch.Tensor):
    """Convert tensor to numpy array (moves to CPU if needed).
    
    Args:
        tensor: torch.Tensor.
        
    Returns:
        numpy array.
    """
    return tensor.detach().cpu().numpy()


def tensors_to_cpu(obj):
    """Recursively convert all tensors in object to CPU.
    
    Handles nested structures (dicts, lists, tuples) and converts only tensors.
    All saved state should be CPU-only to enable seamless device portability.
    
    Args:
        obj: Object that may contain torch.Tensor instances (dict, list, tuple, or tensor).
        
    Returns:
        Object with all tensors on CPU. Other types returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu() if obj.device.type != 'cpu' else obj
    elif isinstance(obj, dict):
        return {key: tensors_to_cpu(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(tensors_to_cpu(item) for item in obj)
    else:
        # Non-tensor objects (numbers, strings, etc.) are returned unchanged
        return obj


def tensors_to_device(obj, device: Optional[torch.device] = None):
    """Recursively convert all tensors in object to specified device.
    
    Used to restore tensors to the active device after loading from CPU state.
    Handles nested structures (dicts, lists, tuples) and converts only tensors.
    
    Args:
        obj: Object that may contain torch.Tensor instances.
        device: Target device. If None, uses current device.
        
    Returns:
        Object with all tensors on target device. Other types returned unchanged.
    """
    target_device = device or _device
    
    if isinstance(obj, torch.Tensor):
        return obj.to(device=target_device) if obj.device != target_device else obj
    elif isinstance(obj, dict):
        return {key: tensors_to_device(value, device=target_device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(tensors_to_device(item, device=target_device) for item in obj)
    else:
        # Non-tensor objects are returned unchanged
        return obj


def reset() -> None:
    """Reset device manager to CPU (useful for testing).
    
    Warning:
        This clears the device state. Use with caution in tests only.
    """
    global _device, _use_gpu
    _device = torch.device('cpu')
    _use_gpu = False
    torch.set_default_dtype(_default_dtype)
