"""
Gym‑compatible wrapper for SnakeGame.
Supports rectangular grid maze.
"""
from typing import Optional
import numpy as np
from snake_game.game import SnakeGame


class SnakeEnv:
    """Gym‑compatible wrapper for SnakeGame with maze."""

    def __init__(self, grid_width: int = 40, grid_height: int = 22, cell_size: int = 30,
                 max_steps: Optional[int] = None):
        self.game = SnakeGame(grid_width, grid_height, cell_size)
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # Max steps to prevent loops
        self.max_steps = max_steps or (grid_width * grid_height * 2)
        self._steps = 0

        # Gym‑like metadata
        self.action_space = type("Discrete", (), {"n": 3})

    def reset(self):
        self._steps = 0
        return self.game.reset()

    def step(self, action: int):
        self._steps += 1
        state, reward, done = self.game.step(action)
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
        self.reset()
        self.render()
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
                        _, _, done, _ = self.step(action)
                        if done:
                            print(f"Game over! Score: {self.game.score}")
                            self.reset()
                    self.render()
        pygame.quit()
