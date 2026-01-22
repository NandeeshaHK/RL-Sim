"""Abstract base class for all RL agents."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """Abstract base class defining the interface for all RL agents.
    
    All agents must implement these methods to ensure compatibility
    with the training loop and GUI visualization.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        """Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space.
            action_dim: Number of discrete actions.
            device: Torch device (CPU or CUDA).
            config: Configuration dictionary with hyperparameters.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.config = config
        
        # Training state
        self.training_step: int = 0
        self.episode_count: int = 0
        
        # Metrics storage (for visualization)
        self._last_loss: float = 0.0
        self._last_q_values: np.ndarray | None = None

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action given the current state.
        
        Args:
            state: Current environment state.
            training: If True, may use exploration. If False, use greedy policy.
            
        Returns:
            Selected action index.
        """
        pass

    @abstractmethod
    def train_step(self) -> dict[str, float]:
        """Perform a single training step.
        
        Returns:
            Dictionary of training metrics (loss, td_error, etc.).
        """
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state after action.
            done: Whether episode terminated.
        """
        pass

    @abstractmethod
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state.
        
        Args:
            state: Current environment state.
            
        Returns:
            Array of Q-values for each action.
        """
        pass

    @abstractmethod
    def get_network(self) -> nn.Module:
        """Get the main Q-network for visualization.
        
        Returns:
            The neural network module.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to the checkpoint file.
        """
        pass

    def get_v_value(self, state: np.ndarray) -> float:
        """Get the state value V(s) = max_a Q(s, a).
        
        Args:
            state: Current environment state.
            
        Returns:
            State value.
        """
        q_values = self.get_q_values(state)
        return float(np.max(q_values))

    def get_metrics(self) -> dict[str, float]:
        """Get current training metrics for visualization.
        
        Returns:
            Dictionary of current metrics.
        """
        return {
            "loss": self._last_loss,
            "training_step": float(self.training_step),
            "episode_count": float(self.episode_count),
        }

    def update_episode_count(self) -> None:
        """Increment the episode counter."""
        self.episode_count += 1

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Current exploration rate (for epsilon-greedy agents)."""
        pass

    def can_train(self) -> bool:
        """Check if agent has enough samples to start training.
        
        Returns:
            True if training can begin.
        """
        return True  # Override in subclasses
