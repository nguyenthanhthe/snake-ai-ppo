"""
Watch a trained PPO agent play Snake.

Usage:
    conda activate snake-ai-ppo
    python test.py [--model models/ppo_snake_final.pt]
"""
import argparse
import torch
from config import Config
from snake_game.env import SnakeEnv
from agent.model import ActorCritic


def test(model_path: str, grid_size: int, episodes: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(11, 3).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()

    env = SnakeEnv(grid_size=grid_size, cell_size=40)

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            s = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = net(s.unsqueeze(0))
                action = torch.distributions.Categorical(logits=logits).sample().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {ep}: score {env.game.score}, total reward {total_reward:.0f}")

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ppo_snake_final.pt")
    p.add_argument("--grid", type=int, default=Config().grid_size)
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    test(args.model, args.grid, args.episodes)
