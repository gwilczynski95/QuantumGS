from abc import ABC, abstractmethod

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn


def positional_encoding(x: torch.Tensor, num_frequencies: int = 6) -> torch.Tensor:
    """Applies positional encoding to the input tensor."""
    encoded = [x]
    frequencies = 2.0**torch.arange(num_frequencies, device=x.device)
    for freq in frequencies:
        encoded.append(torch.sin(x * freq))
        encoded.append(torch.cos(x * freq))
    return torch.cat(encoded, dim=-1)


class SharedEncoder(nn.Module, ABC):
    """
    An abstract parent class for all shared direction encoders.

    It defines the common structure: a feature extraction step followed by an MLP
    head that outputs a 3D vector, which is then passed through tanh.

    Child classes MUST implement the `_get_features` method.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 3)
        )

    @abstractmethod
    def _get_features(self, directions: torch.Tensor) -> torch.Tensor:
        """
        This method must be implemented by all child classes.
        It takes the raw 3D direction vectors and returns the processed feature
        tensor that will be fed into the `mlp_head`.
        """
        raise NotImplementedError("Child classes must implement the `_get_features` method.")

    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        The common forward pass for all encoders.
        
        Args:
            directions (torch.Tensor): Input view directions, shape [N, 3].
        
        Returns:
            torch.Tensor: Encoded 3D feature vectors in range (-1, 1), shape [N, 3].
        """
        # 1. Get the unique features from the child implementation.
        features = self._get_features(directions)
        
        # 2. Process features through the common MLP head.
        mlp_output = self.mlp_head(features)
        
        # 3. Apply final activation to ensure the (-1, 1) range.
        return torch.tanh(mlp_output)


class SharedIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        return directions


class SharedMLPEncoder(SharedEncoder):
    def __init__(self, hidden_dim: int = 32):
        # The input to the MLP head is just the raw 3D direction.
        super().__init__(input_dim=3, hidden_dim=hidden_dim)

    def _get_features(self, directions: torch.Tensor) -> torch.Tensor:
        return directions


class SharedPositionalMLPEncoder(SharedEncoder):
    def __init__(self, num_frequencies: int = 4, hidden_dim: int = 32):
        self.num_frequencies = num_frequencies
        input_dim = 3 + 3 * 2 * self.num_frequencies
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)

    def _get_features(self, directions: torch.Tensor) -> torch.Tensor:
        return positional_encoding(directions, num_frequencies=self.num_frequencies)


class SharedRFFMLPEncoder(SharedEncoder):
    def __init__(self, rff_output_dim: int = 32, hidden_dim: int = 32, 
                 sigma: float = 5.0, is_learnable: bool = False):
        if rff_output_dim % 2 != 0:
            raise ValueError("rff_output_dim must be an even number.")
        
        super().__init__(input_dim=rff_output_dim, hidden_dim=hidden_dim)
        
        # The B matrix projects 3D input to (rff_output_dim / 2)
        b_matrix = torch.randn(3, rff_output_dim // 2) * sigma
        if is_learnable:
            self.b_matrix = nn.Parameter(b_matrix)
        else:
            self.register_buffer('b_matrix', b_matrix)
            
    def _get_features(self, directions: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * (directions @ self.b_matrix)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class SharedHashGridMLPEncoder(SharedEncoder):
    def __init__(self, hidden_dim: int = 32, config: dict = None):
        if config is None:
            config = {
                "otype": "HashGrid", 
                "n_levels": 4, 
                "n_features_per_level": 4,
                "log2_hashmap_size": 15, 
                "base_resolution": 16, 
                "per_level_scale": 1.5
            }
        tcnn_encoder_local = tcnn.Encoding(3, encoding_config=config)
        input_dim_for_super = tcnn_encoder_local.n_output_dims
        del tcnn_encoder_local
        super().__init__(input_dim=input_dim_for_super, hidden_dim=hidden_dim)
        self.tcnn_encoder = tcnn.Encoding(3, encoding_config=config)

    def _get_features(self, directions: torch.Tensor) -> torch.Tensor:
        return self.tcnn_encoder(directions).float()

class SharedHashGrid2DMLPEncoder(SharedEncoder):
    def __init__(self, hidden_dim: int = 32, config: dict = None):
        if config is None:
            config = {
                "otype": "HashGrid", 
                "n_levels": 8,
                "n_features_per_level": 2,
                "log2_hashmap_size": 17,
                "base_resolution": 16,
                "per_level_scale": 1.5
            }
        
        tcnn_encoder_local = tcnn.Encoding(2, encoding_config=config)
        input_dim_for_super = tcnn_encoder_local.n_output_dims
        del tcnn_encoder_local
        super().__init__(input_dim=input_dim_for_super, hidden_dim=hidden_dim)
        self.tcnn_encoder = tcnn.Encoding(2, encoding_config=config)
        
    def _get_features(self, directions: torch.Tensor) -> torch.Tensor:
        eps = 1e-6

        theta = torch.atan2(directions[:, 1], directions[:, 0])
        theta_norm = (theta + torch.pi) / (2 * torch.pi)
        
        phi = torch.acos(directions[:, 2].clamp(-1.0 + eps, 1.0 - eps))
        phi_norm = phi / torch.pi
        
        hash_input = torch.stack([theta_norm, phi_norm], dim=-1)
        return self.tcnn_encoder(hash_input).float()
