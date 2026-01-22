"""Tests for RL algorithms."""

import pytest
import numpy as np
import torch

from src.algorithms.base_agent import BaseAgent
from src.algorithms.double_dqn import DoubleDQNAgent
from src.algorithms.rainbow_dqn import RainbowDQNAgent


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def config() -> dict:
    """Get test configuration."""
    return {
        "training": {
            "gamma": 0.99,
            "learning_rate": 0.001,
            "batch_size": 32,
            "buffer_size": 1000,
            "target_update_freq": 100,
        },
        "double_dqn": {
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.99,
            "tau": 0.005,
        },
        "rainbow": {
            "n_atoms": 11,
            "v_min": -100,
            "v_max": 100,
            "n_steps": 3,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 1000,
            "noisy_std": 0.5,
        },
        "network": {
            "hidden_layers": [32, 32],
            "activation": "relu",
        },
    }


class TestDoubleDQN:
    """Tests for Double DQN agent."""

    def test_init(self, device: torch.device, config: dict) -> None:
        """Test agent initialization."""
        agent = DoubleDQNAgent(8, 4, device, config)
        
        assert agent.state_dim == 8
        assert agent.action_dim == 4
        assert agent.epsilon == 1.0

    def test_select_action(self, device: torch.device, config: dict) -> None:
        """Test action selection."""
        agent = DoubleDQNAgent(8, 4, device, config)
        state = np.random.randn(8).astype(np.float32)
        
        action = agent.select_action(state, training=False)
        
        assert 0 <= action < 4

    def test_store_transition(self, device: torch.device, config: dict) -> None:
        """Test transition storage."""
        agent = DoubleDQNAgent(8, 4, device, config)
        
        state = np.random.randn(8).astype(np.float32)
        next_state = np.random.randn(8).astype(np.float32)
        
        agent.store_transition(state, 0, 1.0, next_state, False)
        
        assert len(agent.buffer) == 1

    def test_train_step(self, device: torch.device, config: dict) -> None:
        """Test training step."""
        agent = DoubleDQNAgent(8, 4, device, config)
        
        # Fill buffer
        for _ in range(50):
            state = np.random.randn(8).astype(np.float32)
            next_state = np.random.randn(8).astype(np.float32)
            agent.store_transition(state, 0, 1.0, next_state, False)
        
        metrics = agent.train_step()
        
        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_get_q_values(self, device: torch.device, config: dict) -> None:
        """Test Q-value retrieval."""
        agent = DoubleDQNAgent(8, 4, device, config)
        state = np.random.randn(8).astype(np.float32)
        
        q_values = agent.get_q_values(state)
        
        assert q_values.shape == (4,)

    def test_epsilon_decay(self, device: torch.device, config: dict) -> None:
        """Test epsilon decay."""
        agent = DoubleDQNAgent(8, 4, device, config)
        initial_epsilon = agent.epsilon
        
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon


class TestRainbowDQN:
    """Tests for Rainbow DQN agent."""

    def test_init(self, device: torch.device, config: dict) -> None:
        """Test agent initialization."""
        agent = RainbowDQNAgent(8, 4, device, config)
        
        assert agent.state_dim == 8
        assert agent.action_dim == 4
        assert agent.epsilon == 0.0  # Rainbow uses noisy nets

    def test_select_action(self, device: torch.device, config: dict) -> None:
        """Test action selection."""
        agent = RainbowDQNAgent(8, 4, device, config)
        state = np.random.randn(8).astype(np.float32)
        
        action = agent.select_action(state)
        
        assert 0 <= action < 4

    def test_get_q_values(self, device: torch.device, config: dict) -> None:
        """Test Q-value retrieval."""
        agent = RainbowDQNAgent(8, 4, device, config)
        state = np.random.randn(8).astype(np.float32)
        
        q_values = agent.get_q_values(state)
        
        assert q_values.shape == (4,)
