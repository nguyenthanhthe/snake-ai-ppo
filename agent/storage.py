"""
Rollout buffer for PPO.

Stores a full episode of (state, action, reward, done, log_prob, value)
then computes advantages with GAE before the update.
"""
import torch
import numpy as np


class RolloutBuffer:
    """Simple list‑based buffer — one episode at a time."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def get(self, device: torch.device):
        return (torch.as_tensor(np.array(self.states), dtype=torch.float32, device=device),
                torch.as_tensor(np.array(self.actions), dtype=torch.long, device=device),
                torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=device),
                torch.as_tensor(np.array(self.dones), dtype=torch.float32, device=device),
                torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device),
                torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device))
