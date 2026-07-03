"""
Vectorized environment — runs N SnakeGame instances in lockstep.

Provides batched step/reset so PPO can collect diverse trajectories
in parallel for stable, efficient training.
"""
import numpy as np
from snake_game.env import SnakeEnv


class VecEnv:
    """N environments running synchronously."""

    def __init__(self, n_envs: int, grid_size: int = 10,
                 cell_size: int = 40, max_steps: int = 200):
        self.n_envs = n_envs
        self.envs = [SnakeEnv(grid_size, cell_size, max_steps)
                     for _ in range(n_envs)]
        self.obs_dim = 11

    def reset(self):
        """Reset all envs.  Returns (n_envs, obs_dim) float32 array."""
        return np.array([e.reset() for e in self.envs], dtype=np.float32)

    def step(self, actions: np.ndarray):
        """
        Step all envs with the given actions (shape: (n_envs,)).

        Returns (states, rewards, dones, infos).
        Done envs are automatically reset — their terminal state is
        still returned so the PPO update sees the transition.
        """
        states, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            s, r, d, info = env.step(int(actions[i]))
            states.append(s)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
            if d:
                env.reset()  # reset silently so next step starts fresh
        return (np.array(states, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)

    def render_one(self, idx: int = 0):
        """Render one specific environment (for the visualizer)."""
        self.envs[idx].render()

    def get_scores(self):
        """Current score of each env (useful for logging)."""
        return [e.score for e in self.envs]

    def close(self):
        for e in self.envs:
            e.close()
