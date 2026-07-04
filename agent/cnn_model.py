"""
CoordConv + Dilated Convs + CBAM Attention CNN Actor‑Critic for maze snake.

Architecture:
  Input (5, H, W) -> CoordConv (7, H, W) -> Conv1 (32) -> Conv2_Dilated (64)
  -> AttentionResBlock (64) -> Conv3 (64) -> Flatten -> MLP Policy + Value

Highly optimized for pathfinding and spatial coordination inside a grid maze.
"""
import torch
import torch.nn as nn
import numpy as np


def _ortho_init(module, gain=np.sqrt(2)):
    """Orthogonal initialization — best practice for RL networks."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CoordConv2d(nn.Module):
    """Adds normalized X and Y coordinate channels to inputs for spatial awareness."""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Create coordinate channels normalized between -1.0 and 1.0
        xx_range = torch.linspace(-1.0, 1.0, width, device=x.device)
        yy_range = torch.linspace(-1.0, 1.0, height, device=x.device)
        
        xx_channel = xx_range.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        yy_channel = yy_range.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        
        return torch.cat([x, xx_channel, yy_channel], dim=1)


class ChannelAttention(nn.Module):
    """Channel attention block for CBAM."""

    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention block for CBAM."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(concat)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out


class AttentionResBlock(nn.Module):
    """Residual Block with Dilated Convolutions and CBAM Attention."""

    def __init__(self, channels):
        super().__init__()
        # Conv with dilation=4 to capture wide pathfinding context
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        return self.relu(out + x)


class CNNActorCritic(nn.Module):
    """Fully custom CoordConv + Dilated + Attention policy network."""

    def __init__(self, grid_height: int, grid_width: int, n_actions: int = 3,
                 hidden_dim: int = 256, in_channels: int = 5):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.in_channels = in_channels

        # Spatial coordinate injection
        self.coord_conv = CoordConv2d(in_channels)
        
        # Strided CNN trunk to expand receptive field and reduce parameter count by 98%
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # downscale to 20x11
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_block = AttentionResBlock(64)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # downscale to 10x6
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.res_block2 = AttentionResBlock(64)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # downscale to 5x3
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Compute flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_height, grid_width)
            h = self.coord_conv(dummy)
            h = self.conv1(h)
            h = self.conv2(h)
            h = self.res_block(h)
            h = self.conv3(h)
            h = self.res_block2(h)
            h = self.conv4(h)
            feat_dim = h.flatten(1).shape[1]

        # Actor head (policy)
        self.actor_trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, n_actions)

        # Critic head (value)
        self.critic_trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)

        # Apply orthogonal initialization
        self.apply(lambda m: _ortho_init(m, gain=np.sqrt(2)))
        _ortho_init(self.policy, gain=0.01)
        _ortho_init(self.value, gain=1.0)

    def _cnn_forward(self, x):
        """Run CoordConv and CNN trunk, return (features, activations_dict)."""
        coords = self.coord_conv(x)
        a1 = self.conv1(coords)
        a2 = self.conv2(a1)
        a3 = self.res_block(a2)
        a4 = self.conv3(a3)
        a5 = self.res_block2(a4)
        a6 = self.conv4(a5)
        flat = a6.flatten(1)
        
        activations = {
            'conv1': a1, 'conv2': a2,
            'res_block': a3, 'conv3': a6,
        }
        return flat, activations

    def forward(self, x: torch.Tensor):
        """x: (B, 5, H, W). Returns (logits, values)."""
        flat, _ = self._cnn_forward(x)
        actor_h = self.actor_trunk(flat)
        logits = self.policy(actor_h)
        critic_h = self.critic_trunk(flat)
        values = self.value(critic_h).squeeze(-1)
        return logits, values

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
