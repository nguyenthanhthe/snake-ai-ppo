"""
PPO (Proximal Policy Optimization) — from scratch.

Implements the clipped surrogate objective with GAE and entropy bonus.
Minimal, no wrappers, no framework dependencies beyond PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from agent.model import ActorCritic
from agent.storage import RolloutBuffer


class PPO:
    """PPO trainer.  Maintains its own network and optimizer."""

    def __init__(self, n_inputs: int, n_actions: int, device: torch.device,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_epochs: int = 4, batch_size: int = 64):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.net = ActorCritic(n_inputs, n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    # ── rollout ──────────────────────────────────────────────────────────

    def get_action(self, state):
        """Sample action from policy.  Returns (action, log_prob, value)."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value = self.net(s.unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, reward, done, log_prob, value):
        self.buffer.add(state, action, reward, done, log_prob, value)

    # ── update ───────────────────────────────────────────────────────────

    def update(self, last_value: float):
        """Compute advantages with GAE and perform PPO update."""
        states, actions, rewards, dones, old_log_probs, values = \
            self.buffer.get(self.device)

        n = len(states)
        if n < 4:  # too short for meaningful GAE; skip
            self.buffer.clear()
            return

        # Append the bootstrap value for GAE computation
        values_all = torch.cat([values, torch.tensor([last_value], device=self.device)])

        # GAE: δ_t = r_t + γ V(s_{t+1}) * (1 - done_t) - V(s_t)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_all[t + 1] * (1 - dones[t]) - values_all[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Normalise advantages (stabilises training)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()  # std is 0, avoid division

        # Mini‑batch PPO update
        indices = torch.randperm(n)
        for _ in range(self.n_epochs):
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_s = states[idx]
                batch_a = actions[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]
                batch_old_log = old_log_probs[idx]

                logits, values_pred = self.net(batch_s)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_a)
                entropy = dist.entropy().mean()

                # Ratio: π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(log_probs - batch_old_log)

                # Clipped surrogate
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values_pred, batch_ret)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()
