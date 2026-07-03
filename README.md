# 🐍 Snake AI — Deep Reinforcement Learning (PPO from scratch)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep RL agent plays **Snake** using **Proximal Policy Optimization (PPO)** — implemented **from scratch** with PyTorch.
Inspired by [Code Bullet](https://youtu.be/3bhP7zuiFmQ), [Jack of Some](https://www.youtube.com/watch?v=i0Pkgtbh1xw), and [Alex Petrenko](https://www.youtube.com/watch?v=bh_5aIqVTUY).

---

## 🎯 Milestones

| Grid   | Target     | Status |
|:-------|:-----------|:-------|
| 6×6    | Score 35   | ⏳     |
| 10×10  | Score 99   | ⏳     |
| 20×20  | Score 399  | 🎯     |

---

## 🧠 Architecture

```
SnakeGame (Pygame)          → 11‑dim feature vector
       ↓
Actor‑Critic (MLP 128×2)   → action logits, state value
       ↓
PPO (GAE + clipped obj.)   → update policy
```

**Input**: 11 features (danger, direction, food-relative)  
**Actions**: STRAIGHT / TURN_LEFT / TURN_RIGHT  
**PPO**: GAE(λ=0.95), clip ε=0.2, entropy bonus

---

## 📁 Structure

```
snake-ai-ppo/
├── snake_game/
│   ├── game.py          # Core game (Pygame, grid-based)
│   └── env.py           # Gym‑compatible wrapper
├── agent/
│   ├── model.py         # Actor‑Critic (shared trunk)
│   ├── ppo.py           # PPO algorithm (GAE + clipped surrogate)
│   └── storage.py       # Rollout buffer
├── config.py            # Hyperparameters
├── train.py             # Training loop
├── test.py              # Watch trained agent
├── requirements.txt
└── README.md
```

---

## 🚀 Quick start

```bash
# 1. Activate env
conda activate snake-ai-ppo

# 2. Train (Ctrl+C to stop)
python train.py

# 3. Watch the agent (after training)
python test.py --model models/ppo_snake_final.pt
```

---

## 🛠 Setup (new env)

```bash
mamba create -n snake-ai-ppo python=3.12 numpy matplotlib -c conda-forge -y
mamba install -n snake-ai-ppo pygame -c conda-forge -y
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
```

---

## 📚 References

- [Code Bullet — AI learns to play Snake](https://youtu.be/3bhP7zuiFmQ)
- [Jack of Some — Neural Network Learns Snake with DQN & A2C](https://www.youtube.com/watch?v=i0Pkgtbh1xw)
- [Alex Petrenko — Advantage Actor-Critic solves 6×6 Snake](https://www.youtube.com/watch?v=bh_5aIqVTUY)
- [PPO paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

---

## 📄 License

MIT
