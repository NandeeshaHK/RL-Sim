"""Training loop manager with GUI integration."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import gymnasium as gym

from src.algorithms.base_agent import BaseAgent
from src.algorithms.double_dqn import DoubleDQNAgent
from src.algorithms.rainbow_dqn import RainbowDQNAgent
from src.core.config import ConfigManager
from src.core.metrics import MetricsTracker
from src.utils.cuda_manager import CUDAManager


class Trainer:
    """Manages the training loop with GUI callbacks.
    
    Supports pause/resume, speed control, and real-time updates.
    """

    def __init__(
        self,
        config: ConfigManager,
        on_step: Callable[..., None] | None = None,
        on_episode_end: Callable[..., None] | None = None,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            config: Configuration manager.
            on_step: Callback for each training step.
            on_episode_end: Callback for episode end.
        """
        self.config = config
        self.on_step = on_step
        self.on_episode_end = on_episode_end
        
        # CUDA setup
        cuda_config = config.cuda
        self.cuda_manager = CUDAManager(
            max_memory_gb=cuda_config.get("max_vram_gb", 3.5),
            enabled=cuda_config.get("enabled", True),
        )
        self.device = self.cuda_manager.device
        
        # Environment
        self.env: gym.Env | None = None
        self.state_dim = 8  # Lunar Lander
        self.action_dim = 4
        
        # Agent
        self.agent: BaseAgent | None = None
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # Training state
        self.is_running = False
        self.is_paused = False
        self.speed_multiplier = 1
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0

    def setup(self, algorithm: str | None = None) -> None:
        """Setup environment and agent.
        
        Args:
            algorithm: Algorithm name. Uses config default if None.
        """
        algorithm = algorithm or self.config.algorithm
        
        # Create environment
        self.env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
        )
        
        # Create agent
        agent_config = self.config.config
        
        if algorithm == "rainbow":
            self.agent = RainbowDQNAgent(
                self.state_dim,
                self.action_dim,
                self.device,
                agent_config,
            )
        else:
            self.agent = DoubleDQNAgent(
                self.state_dim,
                self.action_dim,
                self.device,
                agent_config,
            )
        
        self.metrics.reset()

    def train_step(self) -> dict[str, Any]:
        """Execute a single training step.
        
        Returns:
            Dictionary of step metrics.
        """
        if self.env is None or self.agent is None:
            raise RuntimeError("Trainer not setup. Call setup() first.")
        
        # Initialize episode if needed
        if self.current_step == 0:
            state, _ = self.env.reset()
            self._current_state = state
            self._episode_reward = 0.0
        
        # Select and execute action
        action = self.agent.select_action(self._current_state)
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        # Store transition
        self.agent.store_transition(
            self._current_state, action, reward, next_state, done
        )
        
        # Train
        train_metrics = {}
        if self.agent.can_train():
            train_metrics = self.agent.train_step()
        
        # Get values for visualization
        q_values = self.agent.get_q_values(self._current_state)
        v_value = self.agent.get_v_value(self._current_state)
        
        # Record metrics
        self.metrics.record_step(
            reward=reward,
            q_value=q_values[action],
            loss=train_metrics.get("loss"),
            v_value=v_value,
        )
        
        # Update state
        self._episode_reward += reward
        self._current_state = next_state
        self.current_step += 1
        self.total_steps += 1
        
        # Get render frame
        frame = self.env.render()
        
        step_info = {
            "episode": self.current_episode,
            "timestep": self.current_step,
            "total_timestep": self.total_steps,
            "reward": self._episode_reward,
            "epsilon": self.agent.epsilon,
            "q_values": q_values,
            "v_value": v_value,
            "state": self._current_state,
            "frame": frame,
            "loss": train_metrics.get("loss", 0.0),
        }
        
        # Call step callback
        if self.on_step is not None:
            self.on_step(**step_info)
        
        # Handle episode end
        if done:
            self._end_episode()
        
        return step_info

    def _end_episode(self) -> None:
        """Handle episode completion."""
        if self.agent is None:
            return
        
        # Decay epsilon for Double DQN
        if hasattr(self.agent, "decay_epsilon"):
            self.agent.decay_epsilon()
        
        self.agent.update_episode_count()
        
        # Get episode metrics
        episode_metrics = self.metrics.end_episode(self.agent.epsilon)
        
        # Call episode callback
        if self.on_episode_end is not None:
            self.on_episode_end(episode_metrics)
        
        # Reset for next episode
        self.current_episode += 1
        self.current_step = 0

    def start(self) -> None:
        """Start training."""
        self.is_running = True
        self.is_paused = False

    def pause(self) -> None:
        """Pause training."""
        self.is_paused = True

    def resume(self) -> None:
        """Resume training."""
        self.is_paused = False

    def stop(self) -> None:
        """Stop training."""
        self.is_running = False
        self.is_paused = False

    def reset(self) -> None:
        """Reset training state."""
        self.stop()
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0
        self.metrics.reset()
        
        if self.agent is not None:
            self.setup()

    def set_speed(self, multiplier: int) -> None:
        """Set training speed.
        
        Args:
            multiplier: Speed multiplier (0 for max speed).
        """
        self.speed_multiplier = multiplier

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training checkpoint.
        
        Args:
            path: Save path.
        """
        if self.agent is None:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.agent.save(str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training checkpoint.
        
        Args:
            path: Checkpoint path.
        """
        if self.agent is None:
            return
        
        self.agent.load(str(path))
        self.current_episode = self.agent.episode_count
        self.total_steps = self.agent.training_step

    def close(self) -> None:
        """Clean up resources."""
        if self.env is not None:
            self.env.close()
