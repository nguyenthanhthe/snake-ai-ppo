"""
Grid observation builder — converts SnakeGame to 5‑channel image.
Supports rectangular grid shapes and full maze layout.

Channels:
  0 → head       (1 pixel)
  1 → body       (gradient: 1.0 near head → 0.2 near tail — encodes ordering)
  2 → food       (1 pixel)
  3 → walls      (maze wall layout + border)
  4 → direction  (arrow pattern — encodes which way the snake is facing)
"""
import numpy as np


def make_grid_obs(head, body, food, grid_height: int, grid_width: int, dir_idx: int = 1,
                  maze: np.ndarray = None):
    """
    Return (5, grid_height, grid_width) float32 array.

    Parameters:
      head        — (x, y) of head
      body        — iterable of (x, y) segments (including head)
      food        — (x, y) of food
      grid_height — rows
      grid_width  — columns
      dir_idx     — direction index (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
      maze        — 2D numpy array of wall layout (0=path, 1=wall)
    """
    obs = np.zeros((5, grid_height, grid_width), dtype=np.float32)

    # Channel 0: Head
    hx, hy = head[0], head[1]
    if 0 <= hx < grid_width and 0 <= hy < grid_height:
        obs[0, hy, hx] = 1.0

    # Channel 1: Body with gradient (head=1.0, tail=0.2)
    body_list = list(body)
    n_seg = len(body_list)
    for i, seg in enumerate(body_list):
        sx, sy = seg[0], seg[1]
        if 0 <= sx < grid_width and 0 <= sy < grid_height:
            intensity = 0.2 + 0.8 * (i / max(n_seg - 1, 1))
            obs[1, sy, sx] = intensity

    # Channel 2: Food
    obs[2, food[1], food[0]] = 1.0

    # Channel 3: Walls
    if maze is not None:
        obs[3] = maze.astype(np.float32)
    else:
        # Fallback to borders if no maze provided
        obs[3, 0, :] = 1.0
        obs[3, -1, :] = 1.0
        obs[3, :, 0] = 1.0
        obs[3, :, -1] = 1.0

    # Channel 4: Direction encoding — arrow/gradient from head
    _DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # UP, RIGHT, DOWN, LEFT
    if 0 <= hx < grid_width and 0 <= hy < grid_height:
        dx, dy = _DIRS[dir_idx]
        obs[4, hy, hx] = 1.0
        # Mark 2 cells in front of head with decreasing intensity
        for step in range(1, 3):
            nx, ny = hx + dx * step, hy + dy * step
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                obs[4, ny, nx] = 1.0 - step * 0.3

    return obs
