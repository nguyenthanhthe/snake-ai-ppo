"""
Gym‑compatible wrapper for SnakeGame.

State space:  Box(11,)   — 11 feature dimensions  (see below)
Action space: Discrete(3) — STRAIGHT (0), LEFT (1), RIGHT (2)

11 features:
  0  danger straight ahead
  1  danger on left
  2  danger on right
  3–6 one‑hot current direction  (UP,RIGHT,DOWN,LEFT)
  7–10 food direction relative to head  (LEFT,RIGHT,UP,DOWN)
"""
from typing import Optional
import numpy as np
from snake_game.game import SnakeGame


class SnakeEnv:
    """Minimal Gym‑compatible wrapper."""

    def __init__(self, grid_size: int = 10, cell_size: int = 40,
                 max_steps: Optional[int] = None):
        self.game = SnakeGame(grid_size, cell_size)
        self.grid_size = grid_size
        self.max_steps = max_steps
        self._steps = 0

        # Gym‑like metadata
        self.observation_space = type("Box", (), {
            "shape": (11,), "low": 0.0, "high": 1.0})
        self.action_space = type("Discrete", (), {"n": 3})

    def reset(self):
        self._steps = 0
        return self.game.reset()

    def step(self, action: int):
        self._steps += 1
        state, reward, done = self.game.step(action)
        # Truncate episode if max_steps reached (prevents infinite looping)
        if self.max_steps and self._steps >= self.max_steps:
            done = True
        return state, reward, done, {}

    def render(self, mode: str = "human"):
        if mode == "human":
            self.game.render()

    def close(self):
        self.game.close()

    @property
    def score(self) -> int:
        return self.game.score

    # ── convenience ──────────────────────────────────────────────────────

    def play_human(self):
        """Let a human play with arrow keys — useful for testing."""
        import pygame
        self.game.reset()
        self.game.render()
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    action = None
                    if ev.key == pygame.K_UP:    action = 0  # straight
                    if ev.key == pygame.K_LEFT:  action = 1  # left
                    if ev.key == pygame.K_RIGHT: action = 2  # right
                    if action is not None:
                        _, _, done = self.game.step(action)
                        if done:
                            print(f"Game over! Score: {self.game.score}")
                            self.game.reset()
                    self.game.render()
        pygame.quit()
