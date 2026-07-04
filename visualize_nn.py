"""
Scalable High-Contrast Light-Theme Neural Network Visualizer for Snake RL.
Displays a split-screen with the Pygame game board on the left and 
a glowing MLP neural network activation path on the right.
Features full window resizability (pygame.RESIZABLE) and a clean light theme.
"""
import sys
import argparse
import time
import pygame
import torch
import numpy as np
from collections import deque

# Windows taskbar icon fix: Set current process AppUserModelID
# before initializing pygame, so Windows groups this python instance separately
# and displays the custom Pygame window icon on the taskbar instead of the default python logo.
if sys.platform == 'win32':
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mycompany.snakerl.visualizer.1.0")
    except Exception as e:
        print(f"Failed to set AppUserModelID: {e}")

from snake_game.game import SnakeGame
from agent.model import ActorCritic

# ── Light Theme Color Palette (Clean Apple-like UI) ───────────────────────
BG_COLOR = (245, 245, 247)       # Light off-white
CARD_BG = (255, 255, 255)        # Pure white
GRID_LINE = (235, 235, 240)      # Soft light grey grid
TEXT_DARK = (30, 30, 35)         # Charcoal black (highest contrast!)
TEXT_MUTED = (120, 120, 128)     # Muted grey for inactive text
NODE_INACTIVE = (210, 210, 215)  # Light grey
SYNAPSE_INACTIVE = (235, 235, 240)# Very faint grey for structure
GREY = (180, 180, 185)

# Neon Accents (Vibrant on White)
NEON_RED = (235, 47, 6)
NEON_GREEN = (39, 174, 96)
NEON_BLUE = (41, 128, 185)
NEON_CYAN = (10, 186, 181)
GOLD = (230, 140, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

INPUT_LABELS = [
    "Danger Straight", "Danger Left", "Danger Right",
    "Facing UP", "Facing RIGHT", "Facing DOWN", "Facing LEFT",
    "Food LEFT", "Food RIGHT", "Food UP", "Food DOWN",
    "Geo Food Straight", "Geo Food Left", "Geo Food Right",
    "Safety Straight", "Safety Left", "Safety Right",
    "Tail Straight", "Tail Left", "Tail Right"
]

OUTPUT_LABELS = ["Straight", "Left", "Right"]


def create_game_icon():
    """Create a beautiful 32x32 red apple with a green leaf as the window icon."""
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(surf, (230, 40, 40), (16, 18), 11)
    pygame.draw.circle(surf, (250, 60, 60), (13, 15), 3) # highlight
    pygame.draw.line(surf, (120, 70, 30), (16, 7), (16, 3), 2)
    pygame.draw.ellipse(surf, (46, 204, 113), (16, 1, 9, 6))
    return surf


def get_activations(net, x):
    """Manually feedforward to capture activations of each layer."""
    acts = {}
    acts['input'] = x.cpu().numpy()[0]
    
    h1_raw = net.trunk[0](x)
    h1 = torch.relu(h1_raw)
    acts['hidden_1'] = h1.cpu().numpy()[0]
    
    h2_raw = net.trunk[2](h1)
    h2 = torch.relu(h2_raw)
    acts['hidden_2'] = h2.cpu().numpy()[0]
    
    logits = net.policy(h2)
    probs = torch.softmax(logits, dim=-1)
    acts['output'] = probs.cpu().numpy()[0]
    
    return acts


def render_text_with_shadow(win, text, font, color, pos, light_theme=True):
    """Draw crisp text with soft high-contrast shading for optimal readability."""
    x, y = pos
    # Soft light shadow for light theme, dark shadow for dark theme
    shadow_color = (220, 220, 225) if light_theme else (10, 10, 15)
    shadow = font.render(text, True, shadow_color)
    win.blit(shadow, (x + 1, y + 1))
    
    fg = font.render(text, True, color)
    win.blit(fg, (x, y))


def draw_glow_circle(win, color, center, radius, glow_radius=18, max_alpha=70):
    """Draw a soft glowing aura around an active node using additive blending."""
    glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
    for r in range(glow_radius, radius, -2):
        ratio = (r - radius) / (glow_radius - radius)
        alpha = int(max_alpha * (1.0 - ratio * ratio))
        pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), r)
    win.blit(glow_surf, (center[0] - glow_radius, center[1] - glow_radius))


def draw_nn_visualizer(win, start_x, start_y, width, height, acts, net, font, bold_font, chosen_action):
    """Draw a highly styled neural network visualizer with active flow glows."""
    h1_indices = [int(i * 256 / 12) for i in range(12)]
    h2_indices = [int(i * 256 / 12) for i in range(12)]

    # Node positions
    x_in = start_x + 180
    x_h1 = start_x + 320
    x_h2 = start_x + 440
    x_out = start_x + 550

    y_coords_in = [start_y + 20 + i * 21 for i in range(20)]
    y_coords_h1 = [start_y + 40 + i * 32 for i in range(12)]
    y_coords_h2 = [start_y + 40 + i * 32 for i in range(12)]
    y_coords_out = [start_y + 110 + i * 90 for i in range(3)]

    # Draw Synapses (Lines)
    # Layer 1: Input -> Hidden 1
    w1 = net.trunk[0].weight.data.cpu().numpy()
    for j_idx, j in enumerate(h1_indices):
        for i in range(20):
            weight = w1[j, i]
            val = acts['input'][i]
            abs_w = abs(weight)
            
            if abs(val) > 0.01:
                intensity = max(0, min(1.0, abs_w * abs(val)))
                thickness = min(3, int(intensity * 4) + 1)
                glow = int(intensity * 180)
                color = (46, 120 + glow, 113) if weight > 0 else (120 + glow, 46, 50)
                pygame.draw.line(win, color, (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), thickness)
            elif abs_w > 0.15:
                pygame.draw.line(win, SYNAPSE_INACTIVE, (x_in, y_coords_in[i]), (x_h1, y_coords_h1[j_idx]), 1)

    # Layer 2: Hidden 1 -> Hidden 2
    w2 = net.trunk[2].weight.data.cpu().numpy()
    for k_idx, k in enumerate(h2_indices):
        for j_idx, j in enumerate(h1_indices):
            weight = w2[k, j]
            h1_act = acts['hidden_1'][j]
            abs_w = abs(weight)
            
            if h1_act > 0.05:
                intensity = max(0, min(1.0, abs_w * h1_act))
                thickness = min(3, int(intensity * 4) + 1)
                glow = int(intensity * 180)
                color = (46, 120 + glow, 113) if weight > 0 else (120 + glow, 46, 50)
                pygame.draw.line(win, color, (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), thickness)
            elif abs_w > 0.15:
                pygame.draw.line(win, SYNAPSE_INACTIVE, (x_h1, y_coords_h1[j_idx]), (x_h2, y_coords_h2[k_idx]), 1)

    # Layer 3: Hidden 2 -> Output
    w3 = net.policy.weight.data.cpu().numpy()
    for o in range(3):
        for k_idx, k in enumerate(h2_indices):
            weight = w3[o, k]
            h2_act = acts['hidden_2'][k]
            abs_w = abs(weight)
            
            if h2_act > 0.05:
                intensity = max(0, min(1.0, abs_w * h2_act))
                thickness = min(3, int(intensity * 4) + 1)
                glow = int(intensity * 180)
                color = (46, 120 + glow, 113) if weight > 0 else (120 + glow, 46, 50)
                pygame.draw.line(win, color, (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), thickness)
            elif abs_w > 0.15:
                pygame.draw.line(win, SYNAPSE_INACTIVE, (x_h2, y_coords_h2[k_idx]), (x_out, y_coords_out[o]), 1)

    # Draw Nodes with radial aura glows & dynamic breathing sizes
    # 1. Input Nodes
    for i in range(20):
        val = acts['input'][i]
        is_active = abs(val) > 0.01
        
        if is_active:
            glow = max(0, min(155, int(abs(val) * 150)))
            color = (39, 174, 96) if val > 0 else (219, 68, 85)
            # Breathing active node size (radius 7)
            draw_glow_circle(win, color, (x_in, y_coords_in[i]), 7, glow_radius=17, max_alpha=110)
            pygame.draw.circle(win, color, (x_in, y_coords_in[i]), 7)
            pygame.draw.circle(win, WHITE, (x_in, y_coords_in[i]), 7, 1)
            
            render_text_with_shadow(win, INPUT_LABELS[i], font, TEXT_DARK, (x_in - 170, y_coords_in[i] - 7))
        else:
            # Inactive node is smaller (radius 4)
            pygame.draw.circle(win, NODE_INACTIVE, (x_in, y_coords_in[i]), 4)
            render_text_with_shadow(win, INPUT_LABELS[i], font, TEXT_MUTED, (x_in - 170, y_coords_in[i] - 7))

    # 2. Hidden Layer 1 Nodes
    for j_idx, j in enumerate(h1_indices):
        val = acts['hidden_1'][j]
        is_active = val > 0.05
        
        if is_active:
            glow = max(0, min(155, int(val * 120)))
            color = (41, 128, 185)
            draw_glow_circle(win, color, (x_h1, y_coords_h1[j_idx]), 6, glow_radius=15, max_alpha=95)
            pygame.draw.circle(win, color, (x_h1, y_coords_h1[j_idx]), 6)
            pygame.draw.circle(win, WHITE, (x_h1, y_coords_h1[j_idx]), 6, 1)
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_h1, y_coords_h1[j_idx]), 4)

    # 3. Hidden Layer 2 Nodes
    for k_idx, k in enumerate(h2_indices):
        val = acts['hidden_2'][k]
        is_active = val > 0.05
        
        if is_active:
            glow = max(0, min(155, int(val * 120)))
            color = (41, 128, 185)
            draw_glow_circle(win, color, (x_h2, y_coords_h2[k_idx]), 6, glow_radius=15, max_alpha=95)
            pygame.draw.circle(win, color, (x_h2, y_coords_h2[k_idx]), 6)
            pygame.draw.circle(win, WHITE, (x_h2, y_coords_h2[k_idx]), 6, 1)
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_h2, y_coords_h2[k_idx]), 4)

    # 4. Output Nodes
    for o in range(3):
        prob = acts['output'][o]
        is_chosen = (o == chosen_action)
        
        if is_chosen:
            color = GOLD
            draw_glow_circle(win, GOLD, (x_out, y_coords_out[o]), 10, glow_radius=22, max_alpha=120)
            pygame.draw.circle(win, GOLD, (x_out, y_coords_out[o]), 10)
            pygame.draw.circle(win, WHITE, (x_out, y_coords_out[o]), 11, 2)
            
            render_text_with_shadow(win, f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", bold_font, GOLD, (x_out + 20, y_coords_out[o] - 8))
        else:
            pygame.draw.circle(win, NODE_INACTIVE, (x_out, y_coords_out[o]), 7)
            render_text_with_shadow(win, f"{OUTPUT_LABELS[o]}: {prob*100:.1f}%", font, TEXT_MUTED, (x_out + 20, y_coords_out[o] - 8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_snake_upd100.pt", help="Path to trained PPO model")
    parser.add_argument("--width", type=int, default=40, help="Grid width")
    parser.add_argument("--height", type=int, default=22, help="Grid height")
    parser.add_argument("--delay", type=int, default=120, help="Step delay in ms")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Actor-Critic network
    net = ActorCritic(n_inputs=20, n_actions=3, hidden_dim=256).to(device)
    try:
        net.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Successfully loaded model weights from {args.model}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
    net.eval()

    # Game initialization
    pygame.init()
    
    # Programmatic window icon setup
    icon = create_game_icon()
    pygame.display.set_icon(icon)

    # Clean dimensions
    cell_size = 18
    game_w = args.width * cell_size
    game_h = args.height * cell_size
    nn_panel_w = 700
    
    virtual_w = game_w + nn_panel_w
    virtual_h = game_h + 60
    
    # Render onto an internal high-fidelity virtual screen
    virtual_surf = pygame.Surface((virtual_w, virtual_h))
    
    # Actual display window is RESIZABLE
    win = pygame.display.set_mode((virtual_w, virtual_h), pygame.RESIZABLE)
    pygame.display.set_caption("Maze Snake AI - Neural Network Activation Visualizer (Light Theme)")
    clock = pygame.time.Clock()

    # Clean typography setup (Segoe UI fallback)
    try:
        font = pygame.font.SysFont("Segoe UI", 12)
        bold_font = pygame.font.SysFont("Segoe UI", 14, bold=True)
        title_font = pygame.font.SysFont("Segoe UI", 16, bold=True)
    except:
        font = pygame.font.SysFont("Arial", 12)
        bold_font = pygame.font.SysFont("Arial", 14, bold=True)
        title_font = pygame.font.SysFont("Arial", 16, bold=True)

    # Initialize environment with light theme
    game = SnakeGame(grid_width=args.width, grid_height=args.height, cell_size=cell_size, theme="light")
    state = game.reset()
    episode = 1
    best_score = 0

    running = True
    while running:
        # Pygame resizable events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Re-establish window dimensions safely
                win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False

        # Run PPO Agent step
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_t, _, _, _ = net.get_action_and_value(state_t)
            action = action_t.item()
            acts = get_activations(net, state_t)

        # Take action in environment
        state, reward, done = game.step(action)

        # 1. Render all frames onto our virtual screen
        virtual_surf.fill(BG_COLOR)
        
        # Draw game board
        game._screen = virtual_surf
        game._draw_frame()

        # Draw Neural Network Visualizer Panel on the Right
        nn_rect = pygame.Rect(game_w, 0, nn_panel_w, game_h)
        pygame.draw.rect(virtual_surf, CARD_BG, nn_rect)
        
        # Render clean grid background
        for x in range(game_w + 30, virtual_w, 30):
            pygame.draw.line(virtual_surf, GRID_LINE, (x, 0), (x, game_h), 1)
        for y in range(30, game_h, 30):
            pygame.draw.line(virtual_surf, GRID_LINE, (game_w, y), (virtual_w, y), 1)
            
        pygame.draw.line(virtual_surf, GREY, (game_w, 0), (game_w, game_h), 2)
        
        # Section Title with drop shadow
        render_text_with_shadow(virtual_surf, "REAL-TIME NEURAL NETWORK ACTIVATIONS (MLP)", title_font, TEXT_DARK, (game_w + 20, 15))
        
        # Render the NN connections & nodes
        draw_nn_visualizer(virtual_surf, game_w, 30, nn_panel_w, game_h - 40, acts, net, font, bold_font, action)

        # Draw Bottom HUD Status Bar
        hud_rect = pygame.Rect(0, game_h, virtual_w, 60)
        pygame.draw.rect(virtual_surf, BG_COLOR, hud_rect)
        pygame.draw.line(virtual_surf, GREY, (0, game_h), (virtual_w, game_h), 2)

        if game.score > best_score:
            best_score = game.score
            
        hud_text = f"Episode: {episode} | Score: {game.score} | Best Score: {best_score} | Steps: {game._steps_since_food}"
        render_text_with_shadow(virtual_surf, hud_text, title_font, TEXT_DARK, (20, game_h + 20))

        # Legend explanation
        legend_txt = "Theme: Light | Resizable: Drag corners | Green/Red Synapse = Weight | Glowing Circle = Firing"
        render_text_with_shadow(virtual_surf, legend_txt, font, TEXT_MUTED, (virtual_w - 530, game_h + 22))

        # 2. Scale the virtual screen to the resized display window smoothly
        win.fill(BLACK)
        win.blit(pygame.transform.smoothscale(virtual_surf, win.get_size()), (0, 0))
        
        pygame.display.flip()
        
        if done:
            state = game.reset()
            episode += 1
            time.sleep(0.5)
            
        clock.tick(1000 / args.delay)

    pygame.quit()


if __name__ == "__main__":
    main()
