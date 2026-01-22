"""DQN Network architectures including standard and dueling variants."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.networks.noisy_linear import NoisyLinear


class DQNNetwork(nn.Module):
    """Standard DQN network with fully connected layers.
    
    Architecture: Input -> Hidden Layers -> Q-values
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
    ) -> None:
        """Initialize the DQN network.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            hidden_dims: Sizes of hidden layers.
            activation: Activation function ("relu", "tanh", "elu").
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Activation function
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
        }
        self.activation_fn = activations.get(activation, nn.ReLU)
        
        # Build layers
        layers: list[nn.Module] = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values.
        
        Args:
            state: Batch of states [batch_size, state_dim].
            
        Returns:
            Q-values for each action [batch_size, action_dim].
        """
        return self.network(state)

    def get_layer_outputs(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Get outputs of each layer for visualization.
        
        Args:
            state: Input state tensor.
            
        Returns:
            List of activation tensors for each layer.
        """
        outputs = []
        x = state
        
        for layer in self.network:
            x = layer(x)
            if isinstance(layer, (nn.Linear, NoisyLinear)):
                outputs.append(x.detach().clone())
        
        return outputs


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN network separating value and advantage streams.
    
    Architecture:
        Input -> Feature Extractor -> [Value Stream, Advantage Stream] -> Q-values
        
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        use_noisy: bool = False,
        noisy_std: float = 0.5,
    ) -> None:
        """Initialize the Dueling DQN network.
        
        Args:
            state_dim: Dimension of state space.
            action_dim: Number of discrete actions.
            hidden_dims: Sizes of hidden layers for feature extraction.
            use_noisy: Whether to use NoisyLinear layers.
            noisy_std: Standard deviation for noisy layers.
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy
        
        # Feature extraction layers
        feature_layers: list[nn.Module] = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1] if len(hidden_dims) > 1 else hidden_dims):
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*feature_layers)
        feature_dim = prev_dim
        
        # Linear layer factory
        def make_linear(in_f: int, out_f: int) -> nn.Module:
            if use_noisy:
                return NoisyLinear(in_f, out_f, noisy_std)
            return nn.Linear(in_f, out_f)
        
        # Value stream: V(s)
        value_hidden = hidden_dims[-1] if hidden_dims else 64
        self.value_stream = nn.Sequential(
            make_linear(feature_dim, value_hidden),
            nn.ReLU(),
            make_linear(value_hidden, 1),
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            make_linear(feature_dim, value_hidden),
            nn.ReLU(),
            make_linear(value_hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Q-values from value and advantage.
        
        Args:
            state: Batch of states.
            
        Returns:
            Q-values for each action.
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)  # [batch, 1]
        advantage = self.advantage_stream(features)  # [batch, action_dim]
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

    def get_value_advantage(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get separate value and advantage for visualization.
        
        Args:
            state: Batch of states.
            
        Returns:
            Tuple of (value, advantage) tensors.
        """
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value, advantage

    def reset_noise(self) -> None:
        """Reset noise for all noisy layers."""
        if not self.use_noisy:
            return
            
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def get_layer_outputs(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Get outputs of each layer for visualization.
        
        Args:
            state: Input state tensor.
            
        Returns:
            List of activation tensors.
        """
        outputs = []
        
        # Feature layer outputs
        x = state
        for layer in self.feature_layer:
            x = layer(x)
            if isinstance(layer, (nn.Linear, NoisyLinear)):
                outputs.append(x.detach().clone())
        
        # Value stream
        features = x
        v = features
        for layer in self.value_stream:
            v = layer(v)
            if isinstance(layer, (nn.Linear, NoisyLinear)):
                outputs.append(v.detach().clone())
        
        return outputs
