import pygame
from typing import Optional

class PastureRenderer:
    """Pygame-based renderer for the PastureEnv.

    Features:
        - Three pasture rectangles (A,B,C) colored by grass level (green intensity).
        - Hunger bar (red fill proportional to hunger).
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
        self.font = pygame.font.SysFont("arial", 16)
        self.clock = pygame.time.Clock()
        self.max_fps = max_fps
        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0

    def reset(self):
        self.last_action = None
        self.last_reward = 0.0
        self.screen.fill((30, 30, 30))
        pygame.display.flip()

    def draw(self, grass_levels, hunger: float, day: int, last_action: Optional[int], reward: float):
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
            # Text overlay
            text_surface = self.font.render(f"{titles[i]} G={g:.2f}", True, (230, 230, 230))
            self.screen.blit(text_surface, (x + 8, y + 8))

        # Hunger bar
        bar_x = margin
        bar_y = 270
        bar_w = self.window_width - 2 * margin
        bar_h = 30
        pygame.draw.rect(self.screen, (90, 90, 90), (bar_x, bar_y, bar_w, bar_h))
        hunger_w = int(bar_w * hunger)
        pygame.draw.rect(self.screen, (200, 50, 50), (bar_x, bar_y, hunger_w, bar_h))
        pygame.draw.rect(self.screen, (230, 230, 230), (bar_x, bar_y, bar_w, bar_h), 2)
        hunger_text = self.font.render(f"Cow Hunger = {hunger:.2f}", True, (240, 240, 240))
        self.screen.blit(hunger_text, (bar_x + 8, bar_y + 5))

        # Status text
        action_map = {0: "A", 1: "B", 2: "C", None: "-"}
        status_lines = [
            f"Day: {day}",
            f"Last Action: {action_map.get(self.last_action, '-')}",
            f"Last Reward: {self.last_reward:.2f}",
            "Objective: sustain grass & reduce hunger"
        ]
        for idx, line in enumerate(status_lines):
            s = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(s, (margin, 20 + idx * 18))

        pygame.display.flip()
        self.clock.tick(self.max_fps)

    def close(self):
        try:
            pygame.display.quit()
        except Exception:
            pass
        pygame.quit()
