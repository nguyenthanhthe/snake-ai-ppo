"""
PPO (Proximal Policy Optimization) — with best practices.

Improvements:
  - Value function clipping
  - Learning rate annealing (linear decay)
  - Entropy coefficient annealing
  - Orthogonal initialization
  - Activation extraction for CNN visualizer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent.model import ActorCritic
from agent.cnn_model import CNNActorCritic


class PPO:
    """PPO trainer with support for MLP and CNN policies."""

    def __init__(self, obs_shape, n_actions: int, device: torch.device,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_epochs: int = 4, batch_size: int = 64,
                 hidden_dim: int = 256,
                 lr_end: float = 0.0, ent_coef_end: float = 0.001,
                 total_updates: int = 1000):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.ent_coef_start = ent_coef
        self.ent_coef_end = ent_coef_end
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.obs_shape = obs_shape

        # Annealing
        self.lr_start = lr
        self.lr_end = lr_end
        self.total_updates = total_updates
        self._update_count = 0

        # Choose network: CNN for grid input, MLP for features
        if isinstance(obs_shape, tuple):
            grid_h, grid_w = obs_shape[1], obs_shape[2]
            self.net = CNNActorCritic(grid_h, grid_w, n_actions, hidden_dim).to(device)
        else:
            self.net = ActorCritic(obs_shape, n_actions, hidden_dim).to(device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

    def _anneal(self):
        """Linear annealing of LR and entropy coefficient."""
        frac = max(0.0, 1.0 - self._update_count / max(self.total_updates, 1))
        # LR annealing
        new_lr = self.lr_end + frac * (self.lr_start - self.lr_end)
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr
        # Entropy annealing
        self.ent_coef = self.ent_coef_end + frac * (self.ent_coef_start - self.ent_coef_end)

    def get_actions(self, states: torch.Tensor):
        """Batched forward.  Returns (actions, log_probs, values)."""
        with torch.no_grad():
            logits, values = self.net(states)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def evaluate(self, state: np.ndarray) -> tuple:
        """Single‑state evaluation (for test/viz).  Returns (action_probs, value)."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        s = s.unsqueeze(0)  # (1, ...)
        with torch.no_grad():
            logits, value = self.net(s)
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy(), value.item()

    def evaluate_with_activations(self, state: np.ndarray) -> tuple:
        """Like evaluate but also returns intermediate activations for CNN visualizer."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        s = s.unsqueeze(0)
        with torch.no_grad():
            if hasattr(self.net, 'forward_with_activations'):
                logits, value, activations = self.net.forward_with_activations(s)
                probs = torch.softmax(logits, dim=-1)
                # Convert activations to numpy
                np_activations = {}
                for k, v in activations.items():
                    np_activations[k] = v.cpu().numpy()
                return probs.squeeze(0).cpu().numpy(), value.item(), np_activations
            else:
                logits, value = self.net(s)
                probs = torch.softmax(logits, dim=-1)
                return probs.squeeze(0).cpu().numpy(), value.item(), {}

    def update(self, buffer, last_values: torch.Tensor):
        """Compute GAE and perform PPO mini‑batch update with value clipping."""
        self._update_count += 1
        self._anneal()

        states, actions, old_log_probs, advantages, returns = \
            buffer.compute_gae(last_values, self.gamma, self.gae_lambda)

        # Store old values for value clipping
        with torch.no_grad():
            _, old_values = self.net(states)

        n = len(states)
        indices = torch.randperm(n, device=self.device)
        for _ in range(self.n_epochs):
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_s = states[idx]
                batch_a = actions[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                batch_old_log = old_log_probs[idx]
                batch_old_val = old_values[idx]

                logits, values_pred = self.net(batch_s)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value clipping — PPO best practice
                val_clipped = batch_old_val + torch.clamp(
                    values_pred - batch_old_val, -self.clip_eps, self.clip_eps)
                vf_loss1 = (values_pred - batch_ret) ** 2
                vf_loss2 = (val_clipped - batch_ret) ** 2
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        buffer.clear()
