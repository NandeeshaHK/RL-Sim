"""Metrics tracking and aggregation."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    reward: float
    length: int
    epsilon: float
    avg_q_value: float
    max_q_value: float
    avg_loss: float


class MetricsTracker:
    """Track and aggregate training metrics.
    
    Provides moving averages and statistics for visualization.
    """

    def __init__(self, window_sizes: list[int] | None = None) -> None:
        """Initialize the metrics tracker.
        
        Args:
            window_sizes: Sizes for moving average windows.
        """
        self.window_sizes = window_sizes or [10, 50, 100, 500]
        
        # Episode-level metrics
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_q_means: list[float] = []
        self.episode_q_maxs: list[float] = []
        
        # Step-level metrics (last N steps)
        max_steps = max(self.window_sizes) * 10
        self.step_losses: deque[float] = deque(maxlen=max_steps)
        self.step_q_values: deque[float] = deque(maxlen=max_steps)
        self.step_v_values: deque[float] = deque(maxlen=max_steps)
        
        # Current episode accumulator
        self._current_rewards: list[float] = []
        self._current_q_values: list[float] = []
        self._current_losses: list[float] = []

    def record_step(
        self,
        reward: float,
        q_value: float | None = None,
        loss: float | None = None,
        v_value: float | None = None,
    ) -> None:
        """Record metrics for a single step.
        
        Args:
            reward: Reward received.
            q_value: Q-value of selected action.
            loss: Training loss.
            v_value: State value.
        """
        self._current_rewards.append(reward)
        
        if q_value is not None:
            self._current_q_values.append(q_value)
            self.step_q_values.append(q_value)
        
        if loss is not None:
            self._current_losses.append(loss)
            self.step_losses.append(loss)
        
        if v_value is not None:
            self.step_v_values.append(v_value)

    def end_episode(self, epsilon: float = 0.0) -> EpisodeMetrics:
        """End current episode and aggregate metrics.
        
        Args:
            epsilon: Current exploration rate.
            
        Returns:
            Aggregated episode metrics.
        """
        episode = len(self.episode_rewards)
        
        total_reward = sum(self._current_rewards)
        length = len(self._current_rewards)
        avg_q = np.mean(self._current_q_values) if self._current_q_values else 0.0
        max_q = np.max(self._current_q_values) if self._current_q_values else 0.0
        avg_loss = np.mean(self._current_losses) if self._current_losses else 0.0
        
        # Store episode metrics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_q_means.append(avg_q)
        self.episode_q_maxs.append(max_q)
        
        # Reset accumulators
        self._current_rewards.clear()
        self._current_q_values.clear()
        self._current_losses.clear()
        
        return EpisodeMetrics(
            episode=episode,
            reward=total_reward,
            length=length,
            epsilon=epsilon,
            avg_q_value=avg_q,
            max_q_value=max_q,
            avg_loss=avg_loss,
        )

    def get_moving_average(
        self,
        metric: str,
        window: int,
    ) -> float | None:
        """Get moving average for a metric.
        
        Args:
            metric: Metric name ("reward", "q_mean", "loss").
            window: Window size.
            
        Returns:
            Moving average value or None if insufficient data.
        """
        data_map = {
            "reward": self.episode_rewards,
            "q_mean": self.episode_q_means,
            "q_max": self.episode_q_maxs,
            "loss": list(self.step_losses),
            "v_value": list(self.step_v_values),
        }
        
        data = data_map.get(metric, [])
        
        if len(data) < window:
            return None
        
        return float(np.mean(data[-window:]))

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics summary.
        
        Returns:
            Dictionary of statistics.
        """
        stats: dict[str, Any] = {
            "total_episodes": len(self.episode_rewards),
        }
        
        if self.episode_rewards:
            stats["last_reward"] = self.episode_rewards[-1]
            stats["best_reward"] = max(self.episode_rewards)
            stats["avg_reward_100"] = self.get_moving_average("reward", 100)
        
        if self.step_losses:
            stats["avg_loss"] = np.mean(list(self.step_losses)[-100:])
        
        return stats

    def reset(self) -> None:
        """Reset all metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_q_means.clear()
        self.episode_q_maxs.clear()
        self.step_losses.clear()
        self.step_q_values.clear()
        self.step_v_values.clear()
        self._current_rewards.clear()
        self._current_q_values.clear()
        self._current_losses.clear()
