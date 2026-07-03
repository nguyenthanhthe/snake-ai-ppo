"""Quick demo: 200 episodes with live visualizer."""
from config import Config
from train import train

cfg = Config()
cfg.total_episodes = 200
cfg.render_interval = 20  # show game every 20 episodes
cfg.model_dir = "models_demo"
train(cfg)
