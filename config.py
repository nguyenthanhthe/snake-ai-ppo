"""
Hyperparameters & settings for Snake PPO training.

Modify freely — this is the single control panel.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── parallel environments ───────────────────────────────────────
    n_envs: int = 8             # number of parallel games
    n_steps: int = 128          # steps per update per env (rollout length)

    # ── environment ─────────────────────────────────────────────────
    grid_size: int = 10         # 10×10 = max score 99
    cell_size: int = 40         # pixels per cell
    max_steps: int = 300        # truncate episode to avoid infinite loop

    # ── PPO ─────────────────────────────────────────────────────────
    n_inputs: int = 11          # feature vector size
    n_actions: int = 3          # STRAIGHT / LEFT / RIGHT
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64

    # ── training ────────────────────────────────────────────────────
    total_episodes: int = 5000  # stop when this many episodes are done
    log_interval: int = 5       # print stats every N updates
    render_interval: int = 10   # show game window every N updates
    save_interval: int = 50     # save model every N updates
    model_dir: str = "models"

    # ── network ─────────────────────────────────────────────────────
    hidden_dim: int = 128
    n_layers: int = 2
