"""Tests for replay buffers."""

import pytest
import numpy as np
import torch

from src.buffers.replay_buffer import ReplayBuffer
from src.buffers.prioritized_buffer import PrioritizedReplayBuffer


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cpu")


class TestReplayBuffer:
    """Tests for standard replay buffer."""

    def test_init(self, device: torch.device) -> None:
        """Test buffer initialization."""
        buffer = ReplayBuffer(1000, 8, device)
        
        assert len(buffer) == 0
        assert buffer.capacity == 1000

    def test_push(self, device: torch.device) -> None:
        """Test pushing transitions."""
        buffer = ReplayBuffer(1000, 8, device)
        
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        
        buffer.push(state, 0, 1.0, next_state, False)
        
        assert len(buffer) == 1

    def test_sample(self, device: torch.device) -> None:
        """Test sampling."""
        buffer = ReplayBuffer(1000, 8, device)
        
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            buffer.push(state, 0, 1.0, next_state, False)
        
        batch = buffer.sample(32)
        
        assert batch.states.shape == (32, 8)
        assert batch.actions.shape == (32,)
        assert batch.rewards.shape == (32,)

    def test_circular(self, device: torch.device) -> None:
        """Test circular buffer behavior."""
        buffer = ReplayBuffer(10, 8, device)
        
        # Fill beyond capacity
        for _ in range(20):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            buffer.push(state, 0, 1.0, next_state, False)
        
        assert len(buffer) == 10  # Stays at capacity


class TestPrioritizedReplayBuffer:
    """Tests for prioritized replay buffer."""

    def test_init(self, device: torch.device) -> None:
        """Test buffer initialization."""
        buffer = PrioritizedReplayBuffer(1000, 8, device)
        
        assert len(buffer) == 0

    def test_push(self, device: torch.device) -> None:
        """Test pushing with priority."""
        buffer = PrioritizedReplayBuffer(1000, 8, device)
        
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        
        buffer.push(state, 0, 1.0, next_state, False)
        
        assert len(buffer) == 1

    def test_sample(self, device: torch.device) -> None:
        """Test prioritized sampling."""
        buffer = PrioritizedReplayBuffer(1000, 8, device)
        
        # Fill buffer
        for _ in range(100):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            buffer.push(state, 0, 1.0, next_state, False)
        
        batch = buffer.sample(32)
        
        assert batch.states.shape == (32, 8)
        assert batch.weights.shape == (32,)
        assert len(batch.indices) == 32

    def test_update_priorities(self, device: torch.device) -> None:
        """Test priority updates."""
        buffer = PrioritizedReplayBuffer(1000, 8, device)
        
        # Fill buffer
        for _ in range(50):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            buffer.push(state, 0, 1.0, next_state, False)
        
        batch = buffer.sample(16)
        td_errors = np.random.rand(16)
        
        # Should not raise
        buffer.update_priorities(batch.indices, td_errors)
