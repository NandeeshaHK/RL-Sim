"""Rainbow DQN Agent implementation.

Combines six improvements to DQN:
1. Double Q-learning
2. Prioritized Experience Replay
3. Dueling Networks
4. Noisy Networks
5. N-step Returns
6. Distributional RL (C51)
"""

from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.algorithms.base_agent import BaseAgent
from src.algorithms.networks.dqn_networks import DuelingDQNNetwork
from src.buffers.prioritized_buffer import PrioritizedReplayBuffer


class RainbowDQNAgent(BaseAgent):
    """Rainbow DQN agent combining multiple DQN improvements.
    
    Uses distributional RL with C51 algorithm to model the
    distribution of returns instead of just the expectation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        config: dict[str, Any],
    ) -> None:
        """Initialize Rainbow DQN agent.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            device: Torch device.
            config: Configuration dictionary.
        """
        super().__init__(state_dim, action_dim, device, config)
        
        # Extract hyperparameters
        training_cfg = config.get("training", {})
        rainbow_cfg = config.get("rainbow", {})
        network_cfg = config.get("network", {})
        
        self.gamma = training_cfg.get("gamma", 0.99)
        self.lr = training_cfg.get("learning_rate", 0.0001)
        self.batch_size = training_cfg.get("batch_size", 32)
        self.buffer_size = training_cfg.get("buffer_size", 100000)
        self.target_update_freq = training_cfg.get("target_update_freq", 1000)
        
        # Rainbow-specific parameters
        self.n_atoms = rainbow_cfg.get("n_atoms", 51)
        self.v_min = rainbow_cfg.get("v_min", -200)
        self.v_max = rainbow_cfg.get("v_max", 200)
        self.n_steps = rainbow_cfg.get("n_steps", 3)
        self.alpha = rainbow_cfg.get("alpha", 0.6)
        self.beta_start = rainbow_cfg.get("beta_start", 0.4)
        self.beta_frames = rainbow_cfg.get("beta_frames", 100000)
        self.noisy_std = rainbow_cfg.get("noisy_std", 0.5)
        
        # Distributional RL support
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        
        hidden_layers = network_cfg.get("hidden_layers", [64, 64])
        
        # Networks with distributional output
        self.online_net = self._build_network(
            state_dim, action_dim, hidden_layers
        ).to(device)
        
        self.target_net = self._build_network(
            state_dim, action_dim, hidden_layers
        ).to(device)
        
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        
        # Prioritized replay buffer
        self.buffer = PrioritizedReplayBuffer(
            self.buffer_size,
            state_dim,
            device,
            alpha=self.alpha,
            beta_start=self.beta_start,
            beta_frames=self.beta_frames,
        )
        
        # N-step buffer
        self.n_step_buffer: deque = deque(maxlen=self.n_steps)

    def _build_network(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list[int],
    ) -> nn.Module:
        """Build distributional dueling network.
        
        Returns a network that outputs Q-value distributions.
        """
        return DistributionalDuelingNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_layers,
            n_atoms=self.n_atoms,
            noisy_std=self.noisy_std,
        )

    @property
    def epsilon(self) -> float:
        """Rainbow uses noisy networks, not epsilon-greedy."""
        return 0.0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using noisy network (no explicit exploration).
        
        Args:
            state: Current state.
            training: Reset noise if training.
            
        Returns:
            Selected action index.
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            
            if training:
                self.online_net.reset_noise()
            
            dist = self.online_net(state_tensor)  # [1, action_dim, n_atoms]
            q_values = (dist * self.support).sum(dim=2)  # [1, action_dim]
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
        """Store transition with n-step return calculation."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return
        
        # Calculate n-step return
        n_step_return = 0.0
        for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            n_step_return += (self.gamma ** i) * r
            if d:
                break
        
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]
        
        self.buffer.push(
            first_state,
            first_action,
            n_step_return,
            last_next_state,
            last_done,
        )

    def train_step(self) -> dict[str, float]:
        """Perform one training step with distributional RL.
        
        Returns:
            Dictionary with training metrics.
        """
        if not self.buffer.is_ready(self.batch_size):
            return {"loss": 0.0, "q_mean": 0.0}
        
        batch = self.buffer.sample(self.batch_size)
        
        # Reset noise for both networks
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Compute current distributions
        current_dist = self.online_net(batch.states)  # [batch, action, atoms]
        actions = batch.actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
        current_dist = current_dist.gather(1, actions).squeeze(1)  # [batch, atoms]
        
        # Compute target distribution
        with torch.no_grad():
            # Double Q-learning: select action with online, evaluate with target
            next_dist = self.online_net(batch.next_states)
            next_q = (next_dist * self.support).sum(dim=2)
            next_actions = next_q.argmax(dim=1)
            
            target_dist = self.target_net(batch.next_states)
            next_actions = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.n_atoms)
            target_dist = target_dist.gather(1, next_actions).squeeze(1)
            
            # Project target distribution
            target_dist = self._project_distribution(
                target_dist, batch.rewards, batch.dones
            )
        
        # Cross-entropy loss
        loss = -(target_dist * torch.log(current_dist + 1e-8)).sum(dim=1)
        
        # Weight by importance sampling
        weighted_loss = (loss * batch.weights).mean()
        
        # Update priorities
        td_errors = loss.detach().cpu().numpy()
        self.buffer.update_priorities(batch.indices, td_errors)
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.training_step += 1
        self._last_loss = weighted_loss.item()
        
        # Hard update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        # Compute Q-values for metrics
        with torch.no_grad():
            q_values = (self.online_net(batch.states) * self.support).sum(dim=2)
        
        return {
            "loss": weighted_loss.item(),
            "q_mean": q_values.mean().item(),
        }

    def _project_distribution(
        self,
        target_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Project target distribution onto support.
        
        Implements the Categorical DQN projection step.
        """
        batch_size = rewards.size(0)
        
        # Compute projected support
        rewards = rewards.unsqueeze(1)  # [batch, 1]
        dones = dones.unsqueeze(1)
        gamma_n = self.gamma ** self.n_steps
        
        # Tz = r + gamma^n * z (clipped to [v_min, v_max])
        tz = rewards + gamma_n * (1 - dones) * self.support.unsqueeze(0)
        tz = tz.clamp(self.v_min, self.v_max)
        
        # Compute projection indices
        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Handle edge cases
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.n_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability
        projected = torch.zeros_like(target_dist)
        offset = torch.arange(batch_size, device=self.device).unsqueeze(1) * self.n_atoms
        
        projected.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (target_dist * (u.float() - b)).view(-1),
        )
        projected.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (target_dist * (b - l.float())).view(-1),
        )
        
        return projected

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get expected Q-values from distribution."""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            dist = self.online_net(state_tensor)
            q_values = (dist * self.support).sum(dim=2)
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
            "training_step": self.training_step,
            "episode_count": self.episode_count,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_step = checkpoint["training_step"]
        self.episode_count = checkpoint["episode_count"]


class DistributionalDuelingNetwork(nn.Module):
    """Dueling network with distributional output for Rainbow DQN."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        n_atoms: int,
        noisy_std: float = 0.5,
    ) -> None:
        """Initialize distributional dueling network."""
        super().__init__()
        
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        
        from src.algorithms.networks.noisy_linear import NoisyLinear
        
        # Feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )
        
        # Value stream (outputs distribution)
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dims[0], hidden_dims[-1], noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dims[-1], n_atoms, noisy_std),
        )
        
        # Advantage stream (outputs distribution for each action)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dims[0], hidden_dims[-1], noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dims[-1], action_dim * n_atoms, noisy_std),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action value distributions.
        
        Args:
            state: Batch of states.
            
        Returns:
            Probability distributions over returns for each action.
            Shape: [batch_size, action_dim, n_atoms]
        """
        batch_size = state.size(0)
        features = self.feature(state)
        
        value = self.value_stream(features)  # [batch, atoms]
        advantage = self.advantage_stream(features)  # [batch, action * atoms]
        
        value = value.view(batch_size, 1, self.n_atoms)
        advantage = advantage.view(batch_size, self.action_dim, self.n_atoms)
        
        # Combine value and advantage
        q_dist = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        return torch.softmax(q_dist, dim=2)

    def reset_noise(self) -> None:
        """Reset noise for all noisy layers."""
        from src.algorithms.networks.noisy_linear import NoisyLinear
        
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
