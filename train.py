"""
Train PPO agent to play Snake inside a generated grid obstacle course.
Supports headless background training (show_gui = False) and fully custom
spatially-aware CNN architecture.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import numpy as np

from config import Config
from snake_game.vec_env import VecEnv
from agent.ppo import PPO
from agent.storage import RolloutBuffer


def train(cfg: Config):
    # Set headless driver for pygame if gui is disabled
    if not cfg.show_gui:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[Device] Device: {device}  ({gpu_name})")
    print(f"[VecEnv] Envs: {cfg.n_envs} parallel | Grid: {cfg.grid_width}x{cfg.grid_height}")
    print(f"[PPO] Mode: {cfg.obs_mode.upper()} | Hidden: {cfg.hidden_dim} | "
          f"GUI: {'ON' if cfg.show_gui else 'OFF (Background)'}")

    # Parallel envs with rectangular dimensions
    vec_env = VecEnv(cfg.n_envs, cfg.grid_width, cfg.grid_height, cfg.cell_size, cfg.max_steps,
                     obs_mode=cfg.obs_mode)

    # PPO agent with annealing
    obs_shape = vec_env.obs_shape
    agent = PPO(obs_shape=obs_shape, n_actions=cfg.n_actions, device=device,
                lr=cfg.lr, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda,
                clip_eps=cfg.clip_eps, ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
                n_epochs=cfg.n_epochs, batch_size=cfg.batch_size, hidden_dim=cfg.hidden_dim,
                lr_end=cfg.lr_end, ent_coef_end=cfg.ent_coef_end,
                total_updates=cfg.total_updates_estimate)

    # Rollout buffer
    buffer = RolloutBuffer(cfg.n_envs, cfg.n_steps, obs_shape, device)

    # Checkpoints directory
    os.makedirs(cfg.model_dir, exist_ok=True)

    # ── stats ──────────────────────────────────────────────────────
    total_episodes = 0
    best_score = -1
    episode_returns = []
    episode_scores = []
    ep_return_buf = np.zeros(cfg.n_envs, dtype=np.float32)

    start_time = time.time()
    timesteps = 0

    states = vec_env.reset()

    try:
        update_idx = 0
        while total_episodes < cfg.total_episodes:
            update_idx += 1

            # ── collect rollout ──────────────────────────────────
            for step in range(cfg.n_steps):
                s = torch.as_tensor(states, dtype=torch.float32, device=device)
                actions, log_probs, values = agent.get_actions(s)
                next_states, rewards, dones, _ = vec_env.step(actions)

                buffer.store(states, actions, rewards, dones, log_probs, values)
                states = next_states
                timesteps += cfg.n_envs
                ep_return_buf += rewards

                for i in range(cfg.n_envs):
                    if dones[i]:
                        total_episodes += 1
                        episode_returns.append(ep_return_buf[i])
                        episode_scores.append(vec_env.get_scores()[i])
                        ep_return_buf[i] = 0.0

            # ── PPO update ────────────────────────────────────────
            last_s = torch.as_tensor(states, dtype=torch.float32, device=device)
            with torch.no_grad():
                _, last_val = agent.net(last_s)
            agent.update(buffer, last_val)

            current_scores = vec_env.get_scores()
            best_score = max(best_score, max(current_scores))

            # ── console log ───────────────────────────────────────
            if update_idx % cfg.log_interval == 0:
                n = min(len(episode_returns), cfg.n_envs * 2) if episode_returns else 1
                avg_ret = float(np.mean(episode_returns[-n:])) if episode_returns else 0.0
                eps_sec = total_episodes / (time.time() - start_time)
                lr_now = agent.optimizer.param_groups[0]['lr']
                print(f"Upd {update_idx:4d} | ep {total_episodes:5d}/{cfg.total_episodes} | "
                      f"avg_ret {avg_ret:7.2f} | best {best_score:2d} | {eps_sec:.1f} ep/s | "
                      f"lr={lr_now:.2e}")

            # ── save checkpoint ───────────────────────────────────
            if update_idx % cfg.save_interval == 0:
                path = os.path.join(cfg.model_dir, f"ppo_snake_upd{update_idx}.pt")
                torch.save(agent.net.state_dict(), path)
                print(f"[Model] Saved {path}")

    except KeyboardInterrupt:
        print("\n[Stop] Training interrupted by user.")

    # ── final save ──────────────────────────────────────────────
    final_path = os.path.join(cfg.model_dir, "ppo_snake_final.pt")
    torch.save(agent.net.state_dict(), final_path)
    print(f"[Model] Final model saved -> {final_path}")

    print(f"[Result] Best score: {best_score}")
    print(f"[Result] Total episodes: {total_episodes} | Steps: {timesteps}")
    vec_env.close()


if __name__ == "__main__":
    train(Config())
