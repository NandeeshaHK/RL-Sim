"""Double DQN Agent implementation."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.base_agent import BaseAgent
from src.algorithms.networks.dqn_networks import DQNNetwork
from src.buffers.replay_buffer import ReplayBuffer


class DoubleDQNAgent(BaseAgent):
    """Double DQN agent with epsilon-greedy exploration.
    
    Implements Double Q-learning to reduce overestimation bias:
    - Online network selects actions
    - Target network evaluates Q-values
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        """Initialize Double DQN agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            device: Torch device.
            config: Configuration dictionary with hyperparameters.
        """
        super().__init__(state_dim, action_dim, device, config)
        
        # Extract hyperparameters
        training_cfg = config.get("training", {})
        ddqn_cfg = config.get("double_dqn", {})
        network_cfg = config.get("network", {})
        
        self.gamma = training_cfg.get("gamma", 0.99)
        self.lr = training_cfg.get("learning_rate", 0.0001)
        self.batch_size = training_cfg.get("batch_size", 64)
        self.buffer_size = training_cfg.get("buffer_size", 100000)
        self.target_update_freq = training_cfg.get("target_update_freq", 1000)
        
        self.epsilon_start = ddqn_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = ddqn_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = ddqn_cfg.get("epsilon_decay", 0.995)
        self.tau = ddqn_cfg.get("tau", 0.005)
        
        self._epsilon = self.epsilon_start
        
        hidden_layers = network_cfg.get("hidden_layers", [64, 64])
        activation = network_cfg.get("activation", "relu")
        
        # Networks
        self.online_net = DQNNetwork(
            state_dim, action_dim, hidden_layers, activation
        ).to(device)
        
        self.target_net = DQNNetwork(
            state_dim, action_dim, hidden_layers, activation
        ).to(device)
        
        # Initialize target network with online weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(self.buffer_size, state_dim, device)

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state.
            training: Use exploration if True.
            
        Returns:
            Selected action index.
        """
        if training and np.random.random() < self._epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            self._last_q_values = q_values.cpu().numpy().flatten()
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> dict[str, float]:
        """Perform one training step.
        
        Returns:
            Dictionary with 'loss' and 'q_mean' metrics.
        """
        if not self.buffer.is_ready(self.batch_size):
            return {"loss": 0.0, "q_mean": 0.0}
        
        batch = self.buffer.sample(self.batch_size)
        
        # Compute current Q-values
        current_q = self.online_net(batch.states)
        current_q_values = current_q.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: select action with online, evaluate with target
        with torch.no_grad():
            next_actions = self.online_net(batch.next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(batch.next_states).gather(1, next_actions).squeeze(1)
            target_q = batch.rewards + self.gamma * next_q * (1 - batch.dones)
        
        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.training_step += 1
        self._last_loss = loss.item()
        
        # Soft update target network
        if self.training_step % self.target_update_freq == 0:
            self._soft_update()
        
        return {
            "loss": loss.item(),
            "q_mean": current_q_values.mean().item(),
        }

    def _soft_update(self) -> None:
        """Soft update target network parameters."""
        for target_param, online_param in zip(
            self.target_net.parameters(), self.online_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

    def decay_epsilon(self) -> None:
        """Decay exploration rate after episode."""
        self._epsilon = max(self.epsilon_end, self._epsilon * self.epsilon_decay)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.cpu().numpy().flatten()

    def get_network(self) -> nn.Module:
        """Get online network for visualization."""
        return self.online_net

    def can_train(self) -> bool:
        """Check if enough samples to train."""
        return self.buffer.is_ready(self.batch_size)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "training_step": self.training_step,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]

    def get_metrics(self) -> dict[str, float]:
        """Get current metrics for visualization."""
        metrics = super().get_metrics()
        metrics["epsilon"] = self._epsilon
        return metrics
