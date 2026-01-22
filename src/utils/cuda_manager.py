"""CUDA memory management for efficient VRAM usage."""

import gc
from typing import Any

import torch
import torch.nn as nn


class CUDAManager:
    """Manager for CUDA device and memory optimization.
    
    Features:
    - Automatic device selection
    - Memory budget management
    - Periodic cache clearing
    - Batch size optimization
    """

    def __init__(
        self,
        max_memory_gb: float = 3.5,
        enabled: bool = True,
        device_id: int = 0,
    ) -> None:
        """Initialize CUDA manager.
        
        Args:
            max_memory_gb: Maximum VRAM to use (GB).
            enabled: Whether to use CUDA.
            device_id: CUDA device index.
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.enabled = enabled and torch.cuda.is_available()
        self.device_id = device_id
        
        self._step_count = 0
        self._cache_clear_interval = 100
        
        # Select device
        if self.enabled:
            self.device = torch.device(f"cuda:{device_id}")
            # Set memory fraction
            self._setup_memory_limit()
        else:
            self.device = torch.device("cpu")

    def _setup_memory_limit(self) -> None:
        """Configure CUDA memory allocation."""
        if not self.enabled:
            return
        
        try:
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(
                self.device_id
            ).total_memory
            
            # Calculate fraction
            fraction = min(1.0, self.max_memory_bytes / total_memory)
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(
                fraction, self.device_id
            )
        except Exception:
            # Fallback: just use as is
            pass

    def step(self) -> None:
        """Called each training step for memory management."""
        self._step_count += 1
        
        if self._step_count % self._cache_clear_interval == 0:
            self.clear_cache()

    def clear_cache(self) -> None:
        """Clear CUDA cache and run garbage collection."""
        if self.enabled:
            torch.cuda.empty_cache()
        gc.collect()

    def get_memory_stats(self) -> dict[str, float]:
        """Get current VRAM usage statistics.
        
        Returns:
            Dictionary with memory stats in MB.
        """
        if not self.enabled:
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        return {
            "allocated": torch.cuda.memory_allocated(self.device_id) / 1024**2,
            "reserved": torch.cuda.memory_reserved(self.device_id) / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated(self.device_id) / 1024**2,
        }

    def optimize_batch_size(
        self,
        model: nn.Module,
        state_dim: int,
        target_memory_fraction: float = 0.6,
    ) -> int:
        """Estimate optimal batch size for the model.
        
        Args:
            model: Neural network model.
            state_dim: Input state dimension.
            target_memory_fraction: Target VRAM fraction to use.
            
        Returns:
            Recommended batch size.
        """
        if not self.enabled:
            return 64  # CPU default
        
        # Get available memory
        total = torch.cuda.get_device_properties(self.device_id).total_memory
        used = torch.cuda.memory_allocated(self.device_id)
        available = (total - used) * target_memory_fraction
        
        # Estimate memory per sample (rough)
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        sample_memory = state_dim * 4 * 4  # state + gradient overhead
        
        # Conservative estimate
        estimated_batch = int(available / (sample_memory + param_memory / 32))
        
        # Clamp to reasonable range
        return max(16, min(256, estimated_batch))

    def to_device(self, tensor_or_model: Any) -> Any:
        """Move tensor or model to the managed device.
        
        Args:
            tensor_or_model: Tensor or model to move.
            
        Returns:
            Object on target device.
        """
        if hasattr(tensor_or_model, "to"):
            return tensor_or_model.to(self.device)
        return tensor_or_model

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self.enabled

    def __repr__(self) -> str:
        """String representation."""
        if self.enabled:
            return f"CUDAManager(device=cuda:{self.device_id}, max_memory={self.max_memory_gb}GB)"
        return "CUDAManager(device=cpu)"
