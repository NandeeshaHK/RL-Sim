"""Prioritized Experience Replay buffer for Rainbow DQN."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class SumTree:
    """Binary sum tree for efficient priority-based sampling.
    
    Supports O(log n) insertion and sampling operations.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize the sum tree.
        
        Args:
            capacity: Maximum number of leaf nodes (transitions).
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority of a leaf node and propagate change.
        
        Args:
            tree_idx: Index in tree array.
            priority: New priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # Propagate change up to root
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float) -> int:
        """Add a new priority value.
        
        Args:
            priority: Priority for new transition.
            
        Returns:
            Data index for the new entry.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)
        
        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        return data_idx

    def get(self, value: float) -> tuple[int, float, int]:
        """Sample a leaf node based on cumulative sum.
        
        Args:
            value: Random value in [0, total_priority].
            
        Returns:
            Tuple of (tree_index, priority, data_index).
        """
        parent_idx = 0
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total(self) -> float:
        """Total priority sum (root node value)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Maximum priority in leaves."""
        return np.max(self.tree[self.capacity - 1:])


@dataclass
class PrioritizedBatch:
    """Batch of prioritized transitions with importance sampling weights."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    indices: np.ndarray
    weights: torch.Tensor


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer using sum tree.
    
    Implements PER as described in:
    "Prioritized Experience Replay" (Schaul et al., 2015)
    
    Samples transitions with probability proportional to TD-error priority,
    with importance sampling weights to correct for bias.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions.
            state_dim: Dimension of state vectors.
            device: Torch device for tensors.
            alpha: Priority exponent (0 = uniform, 1 = full priority).
            beta_start: Initial importance sampling weight.
            beta_frames: Frames to anneal beta to 1.0.
            epsilon: Small constant to ensure non-zero priority.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        
        # Pre-allocate storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.size = 0
        self.frame_count = 0
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with maximum priority.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode terminated.
        """
        data_idx = self.tree.add(self.max_priority ** self.alpha)
        
        self.states[data_idx] = state
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx] = float(done)
        
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> PrioritizedBatch:
        """Sample a prioritized batch with importance sampling weights.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Prioritized batch with IS weights.
        """
        self.frame_count += 1
        
        # Anneal beta towards 1.0
        beta = min(
            1.0,
            self.beta_start + self.frame_count * (1.0 - self.beta_start) / self.beta_frames
        )
        
        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            
            tree_idx, priority, data_idx = self.tree.get(value)
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        sampling_probs = priorities / self.tree.total
        weights = (self.size * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        return PrioritizedBatch(
            states=torch.from_numpy(self.states[indices]).to(self.device),
            actions=torch.from_numpy(self.actions[indices]).to(self.device),
            rewards=torch.from_numpy(self.rewards[indices]).to(self.device),
            next_states=torch.from_numpy(self.next_states[indices]).to(self.device),
            dones=torch.from_numpy(self.dones[indices]).to(self.device),
            indices=tree_indices,
            weights=torch.from_numpy(weights.astype(np.float32)).to(self.device),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities based on TD-errors.
        
        Args:
            indices: Tree indices of sampled transitions.
            td_errors: Absolute TD-errors for each transition.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size
