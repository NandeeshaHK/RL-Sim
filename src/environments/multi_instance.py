"""Multi-instance environment manager for parallel training."""

from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gymnasium as gym

from src.environments.lunar_lander import LunarLanderWrapper


class MultiInstanceManager:
    """Manages multiple environment instances for parallel execution.
    
    Useful for:
    - Comparing algorithms side-by-side
    - Collecting experience faster
    - Visualizing multiple training runs
    """

    def __init__(
        self,
        max_instances: int = 4,
        render_mode: str = "rgb_array",
    ) -> None:
        """Initialize the multi-instance manager.
        
        Args:
            max_instances: Maximum number of concurrent instances.
            render_mode: Render mode for environments.
        """
        self.max_instances = max_instances
        self.render_mode = render_mode
        
        self.instances: dict[int, LunarLanderWrapper] = {}
        self.states: dict[int, np.ndarray] = {}
        self._next_id = 0

    def create_instance(self) -> int:
        """Create a new environment instance.
        
        Returns:
            Instance ID.
            
        Raises:
            RuntimeError: If max instances reached.
        """
        if len(self.instances) >= self.max_instances:
            raise RuntimeError(
                f"Maximum instances ({self.max_instances}) reached"
            )
        
        instance_id = self._next_id
        self._next_id += 1
        
        env = LunarLanderWrapper(render_mode=self.render_mode)
        self.instances[instance_id] = env
        
        # Initialize state
        state, _ = env.reset()
        self.states[instance_id] = state
        
        return instance_id

    def remove_instance(self, instance_id: int) -> None:
        """Remove an environment instance.
        
        Args:
            instance_id: ID of instance to remove.
        """
        if instance_id in self.instances:
            self.instances[instance_id].close()
            del self.instances[instance_id]
            del self.states[instance_id]

    def reset(self, instance_id: int) -> np.ndarray:
        """Reset a specific instance.
        
        Args:
            instance_id: Instance to reset.
            
        Returns:
            Initial state.
        """
        state, _ = self.instances[instance_id].reset()
        self.states[instance_id] = state
        return state

    def reset_all(self) -> dict[int, np.ndarray]:
        """Reset all instances.
        
        Returns:
            Dictionary mapping instance IDs to initial states.
        """
        for instance_id in self.instances:
            self.reset(instance_id)
        return self.states.copy()

    def step(
        self,
        instance_id: int,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step in a specific instance.
        
        Args:
            instance_id: Instance ID.
            action: Action to take.
            
        Returns:
            Standard gym step tuple.
        """
        env = self.instances[instance_id]
        state, reward, terminated, truncated, info = env.step(action)
        self.states[instance_id] = state
        return state, reward, terminated, truncated, info

    def step_all(
        self,
        actions: dict[int, int],
    ) -> dict[int, tuple[np.ndarray, float, bool, bool, dict]]:
        """Take steps in all instances (parallel).
        
        Args:
            actions: Dictionary mapping instance IDs to actions.
            
        Returns:
            Dictionary mapping instance IDs to step results.
        """
        results: dict[int, tuple] = {}
        
        for instance_id, action in actions.items():
            if instance_id in self.instances:
                results[instance_id] = self.step(instance_id, action)
        
        return results

    def render(self, instance_id: int) -> np.ndarray | None:
        """Get render frame for an instance.
        
        Args:
            instance_id: Instance ID.
            
        Returns:
            RGB frame array or None.
        """
        if instance_id in self.instances:
            return self.instances[instance_id].render()
        return None

    def render_all(self) -> dict[int, np.ndarray]:
        """Get render frames for all instances.
        
        Returns:
            Dictionary mapping instance IDs to frames.
        """
        return {
            instance_id: env.render()
            for instance_id, env in self.instances.items()
        }

    def get_state(self, instance_id: int) -> np.ndarray | None:
        """Get current state for an instance.
        
        Args:
            instance_id: Instance ID.
            
        Returns:
            Current state or None.
        """
        return self.states.get(instance_id)

    @property
    def instance_ids(self) -> list[int]:
        """Get list of active instance IDs."""
        return list(self.instances.keys())

    @property
    def count(self) -> int:
        """Get number of active instances."""
        return len(self.instances)

    def close(self) -> None:
        """Close all instances."""
        for env in self.instances.values():
            env.close()
        self.instances.clear()
        self.states.clear()
