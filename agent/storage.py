"""
Rollout buffer for PPO with parallel environments.

Pre-allocated flat arrays.  Stores `n_steps` steps from `n_envs` envs,
then computes GAE and yields mini‑batches for the PPO update.
"""
import torch
import numpy as np


class RolloutBuffer:
    """Fixed‑size step‑based buffer for vectorized PPO."""

    def __init__(self, n_envs: int, n_steps: int, obs_dim: int,
                 device: torch.device):
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.device = device

        self.states  = torch.zeros(n_steps, n_envs, obs_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, dtype=torch.float32, device=device)
        self.dones   = torch.zeros(n_steps, n_envs, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, dtype=torch.float32, device=device)
        self.values  = torch.zeros(n_steps, n_envs, dtype=torch.float32, device=device)

        self.step = 0

    def store(self, states, actions, rewards, dones, log_probs, values):
        """Store one step from all envs."""
        self.states[self.step]  = torch.as_tensor(states, device=self.device)
        self.actions[self.step] = torch.as_tensor(actions, device=self.device)
        self.rewards[self.step] = torch.as_tensor(rewards, device=self.device)
        self.dones[self.step]   = torch.as_tensor(dones, device=self.device)
        self.log_probs[self.step] = torch.as_tensor(log_probs, device=self.device)
        self.values[self.step]  = torch.as_tensor(values, device=self.device)
        self.step += 1

    def compute_gae(self, last_values: torch.Tensor,
                    gamma: float, gae_lambda: float):
        """
        Compute GAE advantages and returns in‑place.
        last_values: (n_envs,) — value of the final observation.
        """
        n = self.n_steps
        advantages = torch.zeros(n, self.n_envs, device=self.device)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_values if t == n - 1 else self.values[t + 1]
            delta = (self.rewards[t] + gamma * next_val * (1 - self.dones[t])
                     - self.values[t])
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + self.values

        # Flatten for mini‑batch training
        flat_states  = self.states.reshape(-1, self.states.shape[-1])
        flat_actions = self.actions.flatten()
        flat_log     = self.log_probs.flatten()
        flat_adv     = advantages.flatten()
        flat_ret     = returns.flatten()

        # Normalise advantages (stable training)
        if len(flat_adv) > 1:
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        return flat_states, flat_actions, flat_log, flat_adv, flat_ret

    def clear(self):
        self.step = 0
