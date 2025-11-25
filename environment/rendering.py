import pygame
from typing import Optional

class PastureRenderer:
    """Pygame-based renderer for the PastureEnv.

    Features:
        - Three pasture rectangles (A,B,C) colored by grass level (green intensity).
        - Hunger bar and Thirst bar with color gradients.
        - Overlay text: day, last action, last reward, objective.
        - Action highlight: thick yellow border around chosen pasture.
        - Adjustable FPS for slower visualization.
    """

    def __init__(self, window_width: int = 600, window_height: int = 400, max_fps: int = 8):
        pygame.init()
        pygame.display.set_caption("Pasture Allocation Optimizer")
        self.window_width = window_width
        self.window_height = window_height
        self.screen = pygame.display.set_mode((window_width, window_height))
        self.font_small = pygame.font.SysFont("arial", 13)
        self.font = pygame.font.SysFont("arial", 14)
        self.font_large = pygame.font.SysFont("arial", 16)
        self.clock = pygame.time.Clock()
        self.max_fps = max_fps
        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0

    def reset(self):
        self.last_action = None
        self.last_reward = 0.0
        self.screen.fill((30, 30, 30))
        pygame.display.flip()

    def draw(self, grass_levels, hunger: float, day: int, last_action: Optional[int], reward: float, 
             fert_levels=None, thirst: float = None, disease_risk: float = None, rain_flag: float = None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                return
        self.last_action = last_action
        self.last_reward = reward

        self.screen.fill((25, 25, 25))
        margin = 40
        pasture_width = (self.window_width - margin * 4) // 3
        pasture_height = 180

        titles = ["Pasture A", "Pasture B", "Pasture C"]
        fert_levels = fert_levels if fert_levels is not None else [0.7, 0.7, 0.7]
        for i, g in enumerate(grass_levels):
            x = margin + i * (pasture_width + margin)
            y = 60
            # Grass color: interpolate between brown (low) and green (high)
            green_intensity = int(50 + g * 205)
            color = (80, green_intensity, 60)
            pygame.draw.rect(self.screen, color, (x, y, pasture_width, pasture_height))
            # Outline
            outline_color = (200, 200, 200)
            outline_width = 2
            if last_action is not None and i == last_action:
                outline_color = (255, 215, 0)  # gold highlight
                outline_width = 5
            pygame.draw.rect(self.screen, outline_color, (x, y, pasture_width, pasture_height), outline_width)
            # Text overlay: split into two lines to avoid overlap
            text_line1 = self.font_small.render(f"{titles[i]}", True, (240, 240, 240))
            text_line2 = self.font_small.render(f"G={g:.2f} F={fert_levels[i]:.2f}", True, (200, 200, 200))
            self.screen.blit(text_line1, (x + 8, y + 8))
            self.screen.blit(text_line2, (x + 8, y + 26))

        # Hunger bar
        bar_x = margin
        bar_y = 280
        bar_w = self.window_width - 2 * margin
        bar_h = 18
        pygame.draw.rect(self.screen, (70, 70, 70), (bar_x, bar_y, bar_w, bar_h))
        hunger_w = int(bar_w * hunger)
        # Color gradient: low hunger -> green, high -> red
        h_color = (int(50 + 205 * hunger), int(220 - 160 * hunger), 60)
        pygame.draw.rect(self.screen, h_color, (bar_x, bar_y, hunger_w, bar_h))
        pygame.draw.rect(self.screen, (230, 230, 230), (bar_x, bar_y, bar_w, bar_h), 2)
        hunger_text = self.font_small.render(f"Cow Hunger = {hunger:.2f}", True, (240, 240, 240))
        self.screen.blit(hunger_text, (bar_x + 8, bar_y + 1))

        # Thirst bar
        t_val = 0.0 if thirst is None else thirst
        t_y = bar_y + 22
        pygame.draw.rect(self.screen, (70, 70, 70), (bar_x, t_y, bar_w, bar_h))
        thirst_w = int(bar_w * t_val)
        t_color = (60, int(220 - 160 * t_val), int(50 + 205 * t_val))
        pygame.draw.rect(self.screen, t_color, (bar_x, t_y, thirst_w, bar_h))
        pygame.draw.rect(self.screen, (230, 230, 230), (bar_x, t_y, bar_w, bar_h), 2)
        thirst_text = self.font_small.render(f"Cow Thirst = {t_val:.2f}", True, (240, 240, 240))
        self.screen.blit(thirst_text, (bar_x + 8, t_y + 1))

        # Status text - organized cleanly
        action_map = {0: "A", 1: "B", 2: "C", None: "-"}
        risk = 0.0 if disease_risk is None else disease_risk
        rain_txt = "Yes" if rain_flag else "No"
        status_lines = [
            f"Day: {day}  |  Rain: {rain_txt}  |  Last Action: {action_map.get(self.last_action, '-')}",
            f"Disease Risk: {risk:.2f}  |  Reward: {self.last_reward:.2f}",
            f"Goal: Sustain grass & reduce hunger/thirst"
        ]
        for idx, line in enumerate(status_lines):
            s = self.font_small.render(line, True, (220, 220, 220))
            self.screen.blit(s, (margin, 10 + idx * 16))

        pygame.display.flip()
        self.clock.tick(self.max_fps)

    def close(self):
        try:
            pygame.display.quit()
        except Exception:
            pass
        pygame.quit()
