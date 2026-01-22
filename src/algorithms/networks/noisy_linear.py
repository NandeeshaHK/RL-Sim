"""Noisy Linear layer for exploration in Rainbow DQN."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy Linear layer with factorized Gaussian noise.
    
    Implements parameter-space noise for exploration as described in:
    "Noisy Networks for Exploration" (Fortunato et al., 2018)
    
    The layer has learnable parameters mu and sigma for both weights and biases,
    with noise sampled from factorized Gaussian distributions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std_init: float = 0.5,
    ) -> None:
        """Initialize the noisy linear layer.
        
        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            std_init: Initial standard deviation for noise parameters.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise buffers (not learnable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize mu and sigma parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        """Sample new noise for weights and biases using factorized Gaussian."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        """Generate scaled noise using f(x) = sign(x) * sqrt(|x|).
        
        Args:
            size: Size of the noise tensor.
            
        Returns:
            Scaled noise tensor.
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        """String representation for printing."""
        return f"in_features={self.in_features}, out_features={self.out_features}, std_init={self.std_init}"
