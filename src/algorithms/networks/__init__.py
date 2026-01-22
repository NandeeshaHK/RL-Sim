"""Neural network architectures for RL algorithms."""

from src.algorithms.networks.dqn_networks import DQNNetwork, DuelingDQNNetwork
from src.algorithms.networks.noisy_linear import NoisyLinear

__all__ = ["DQNNetwork", "DuelingDQNNetwork", "NoisyLinear"]
