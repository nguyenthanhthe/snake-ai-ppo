"""
Shared Actor‑Critic network for PPO.

Two small MLP heads on a shared trunk:
  - Actor  → policy logits  (action probabilities)
  - Critic → state value V(s)
"""
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Trunk + two heads.  Fully connected, configurable width/depth."""

    def __init__(self, n_inputs: int = 11, n_actions: int = 3,
                 hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()

        # Shared trunk
        layers = [nn.Linear(n_inputs, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        # Heads
        self.policy = nn.Linear(hidden_dim, n_actions)   # logits
        self.value  = nn.Linear(hidden_dim, 1)           # V(s)

    def forward(self, x: torch.Tensor):
        """Return (action_logits, state_value)."""
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """Sample action (or compute log_prob of given action) + value + entropy."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
