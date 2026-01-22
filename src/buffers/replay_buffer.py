"""Standard experience replay buffer."""

from dataclasses import dataclass
from typing import NamedTuple
import random

import numpy as np
import torch


class Transition(NamedTuple):
    """A single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class BatchTransition:
    """Batch of transitions for training."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size circular buffer for storing experience transitions.
    
    Implements uniform random sampling for standard DQN training.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device,
    ) -> None:
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimension of state vectors.
            device: Torch device for batch tensors.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        
        # Pre-allocate numpy arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Batch of transitions as tensors.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return BatchTransition(
            states=torch.from_numpy(self.states[indices]).to(self.device),
            actions=torch.from_numpy(self.actions[indices]).to(self.device),
            rewards=torch.from_numpy(self.rewards[indices]).to(self.device),
            next_states=torch.from_numpy(self.next_states[indices]).to(self.device),
            dones=torch.from_numpy(self.dones[indices]).to(self.device),
        )

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training.
        
        Args:
            batch_size: Required batch size.
            
        Returns:
            True if buffer has at least batch_size samples.
        """
        return self.size >= batch_size
