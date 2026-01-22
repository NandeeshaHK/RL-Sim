"""Lunar Lander environment wrapper with additional features."""

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np


class LunarLanderWrapper(gym.Wrapper):
    """Wrapper for LunarLander-v2 with visualization support.
    
    Features:
    - State normalization
    - Reward shaping options
    - Easy access to state components
    """

    # State indices
    POS_X = 0
    POS_Y = 1
    VEL_X = 2
    VEL_Y = 3
    ANGLE = 4
    ANGULAR_VEL = 5
    LEFT_LEG = 6
    RIGHT_LEG = 7

    # State feature labels
    STATE_LABELS = [
        "position x",
        "position y",
        "velocity x",
        "velocity y",
        "angle",
        "angular velocity",
        "left leg contact",
        "right leg contact",
    ]

    # Action labels
    ACTION_LABELS = [
        "no engine",
        "right engine",
        "down engine",
        "left engine",
    ]

    def __init__(
        self,
        render_mode: str = "rgb_array",
        normalize_state: bool = False,
    ) -> None:
        """Initialize the wrapper.
        
        Args:
            render_mode: Render mode for visualization.
            normalize_state: Whether to normalize state values.
        """
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        super().__init__(env)
        
        self.normalize_state = normalize_state
        
        # Normalization parameters (approximate ranges)
        self._state_mean = np.array([0, 0.5, 0, 0, 0, 0, 0, 0])
        self._state_std = np.array([1.5, 1.5, 5, 5, 3.14, 5, 1, 1])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed.
            options: Additional options.
            
        Returns:
            Tuple of (observation, info).
        """
        state, info = self.env.reset(seed=seed, options=options)
        
        if self.normalize_state:
            state = self._normalize(state)
        
        return state, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        
        if self.normalize_state:
            state = self._normalize(state)
        
        return state, reward, terminated, truncated, info

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values.
        
        Args:
            state: Raw state.
            
        Returns:
            Normalized state.
        """
        return (state - self._state_mean) / self._state_std

    def get_state_dict(self, state: np.ndarray) -> dict[str, float]:
        """Convert state array to labeled dictionary.
        
        Args:
            state: State array.
            
        Returns:
            Dictionary with labeled state values.
        """
        return dict(zip(self.STATE_LABELS, state.tolist()))

    @staticmethod
    def get_action_name(action: int) -> str:
        """Get action name from action index.
        
        Args:
            action: Action index.
            
        Returns:
            Action name string.
        """
        return LunarLanderWrapper.ACTION_LABELS[action]

    @property
    def state_dim(self) -> int:
        """State dimension."""
        return 8

    @property
    def action_dim(self) -> int:
        """Number of actions."""
        return 4
