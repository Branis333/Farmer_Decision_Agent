import math
import random
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from .rendering import PastureRenderer
except Exception:  
    PastureRenderer = None


class PastureEnv(gym.Env):


    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}

    def __init__(self,
                 max_days: int = 30,
                 depletion_rate: float = 0.28,
                 feeding_amount: float = 0.25,
                 regrowth_rate: float = 0.09,
                 base_hunger_increase: float = 0.06,
                 base_thirst_increase: float = 0.06,
                 collapse_patience: int = 3,
                 seed: Optional[int] = None,
                 render_mode: Optional[str] = None,
                 render_fps: int = 8):
        super().__init__()
        self.max_days = max_days
        self.depletion_rate = depletion_rate
        self.feeding_amount = feeding_amount
        self.regrowth_rate = regrowth_rate
        self.base_hunger_increase = base_hunger_increase
        # Missing assignment caused AttributeError when accessing in step()
        self.base_thirst_increase = base_thirst_increase
        self.collapse_patience = collapse_patience
        self.render_mode = render_mode
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Discrete actions: choose one pasture.
        self.action_space = spaces.Discrete(5)
        # Observation: 11 features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)

        # Internal state
        self.grass = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.fertility = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.hunger: float = 0.5
        self.thirst: float = 0.4
        self.disease_risk: float = 0.1
        self.rain_flag: float = 0.0
        self.day: int = 0
        self.low_grass_streak = np.zeros(3, dtype=np.int32) 
        self.grazing_counts = np.zeros(3, dtype=np.int32)
        self.prev_action: Optional[int] = None

        self.renderer = None
        if self.render_mode == "human" and PastureRenderer is not None:
            self.renderer = PastureRenderer(window_width=600, window_height=400, max_fps=render_fps)

    def seed(self, seed: Optional[int] = None):  
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _get_obs(self) -> np.ndarray:
        day_norm = self.day / self.max_days
        return np.array([
            self.grass[0], self.grass[1], self.grass[2],
            self.fertility[0], self.fertility[1], self.fertility[2],
            self.hunger, self.thirst,
            self.disease_risk, self.rain_flag,
            day_norm
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        grazing_balance = (self.grazing_counts / max(1, np.sum(self.grazing_counts))).tolist()
        return {
            "grass_levels": self.grass.copy(),
            "fertility": self.fertility.copy(),
            "hunger": self.hunger,
            "thirst": self.thirst,
            "disease_risk": self.disease_risk,
            "day": self.day,
            "grazing_balance": grazing_balance,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self.grass = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.fertility[:] = 0.7
        self.hunger = 0.5
        self.thirst = 0.4
        self.disease_risk = 0.1
        self.rain_flag = 0.0
        self.day = 0
        self.low_grass_streak[:] = 0
        self.grazing_counts[:] = 0
        self.prev_action = None
        obs = self._get_obs()
        info = self._get_info()
        if self.renderer:
            self.renderer.reset()
            self.renderer.draw(self.grass, self.hunger, self.day, last_action=None, reward=0.0)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        prev_hunger = self.hunger
        prev_thirst = self.thirst

        # Weather/seasonality
        season = 0.5 + 0.5 * math.sin(2 * math.pi * (self.day % 30) / 30.0)
        p_rain = 0.2 + 0.3 * season
        self.rain_flag = 1.0 if self._np_rng.random() < p_rain else 0.0

        # Apply dynamics
        chosen_grass_before = float(self.grass[action]) if action in (0,1,2) else 0.0
        reward = 0.0

        # Reward component: healthy vs overgrazed BEFORE depletion
        if action in (0,1,2):
            if chosen_grass_before >= 0.5:
                reward += 10.0
            elif chosen_grass_before < 0.3:
                reward -= 20.0

        # Base hunger/thirst increases first
        self.hunger += self.base_hunger_increase
        self.thirst += self.base_thirst_increase

        # Action effects
        if action in (0,1,2):
            # Grazing
            self.grass[action] -= self.depletion_rate
            # Feeding efficiency scales with available grass and disease
            feeding_eff = self.feeding_amount * chosen_grass_before * (1.0 - 0.5 * self.disease_risk)
            self.hunger -= feeding_eff
            # Activity increases thirst a bit
            self.thirst += 0.03
        elif action == 3:
            # Rest day: stronger regrowth bonus and fertility recovery, but cows get hungrier
            self.hunger += 0.03
            self.thirst -= 0.02
            self.fertility += 0.03
        elif action == 4:
            # Water action: reduce thirst significantly, slight hunger increase
            self.thirst -= 0.3
            self.hunger += 0.02

        # Resting pasture regrowth
        for i in range(3):
            rain_mult = (1.0 + 0.5 * self.rain_flag)
            base_growth = self.regrowth_rate * self.fertility[i] * rain_mult
            if action in (0,1,2) and i == action:
                # No regrowth on grazed patch
                pass
            else:
                self.grass[i] += base_growth
                # Fertility gentle recovery when resting
                self.fertility[i] += 0.01

        # Fertility loss on overgrazing or repeated grazing
        if action in (0,1,2):
            if chosen_grass_before < 0.3:
                self.fertility[action] -= 0.05
                self.disease_risk += 0.05
            if self.prev_action == action:
                self.disease_risk += 0.03
        else:
            # decay disease slowly when not grazing
            self.disease_risk -= 0.02

        # Hunger/thirst/disease penalty shaping
        hunger_drop = max(-1.0, min(1.0, prev_hunger - self.hunger))
        thirst_drop = max(-1.0, min(1.0, prev_thirst - self.thirst))
        reward += 20.0 * hunger_drop + 15.0 * thirst_drop
        if self.hunger > 0.7:
            reward -= 5.0
        if self.thirst > 0.7:
            reward -= 5.0
        if self.disease_risk > 0.6:
            reward -= 10.0

        # Clip values
        self.grass = np.clip(self.grass, 0.0, 1.0)
        self.fertility = np.clip(self.fertility, 0.1, 1.0)
        self.hunger = float(np.clip(self.hunger, 0.0, 1.0))
        self.thirst = float(np.clip(self.thirst, 0.0, 1.0))
        self.disease_risk = float(np.clip(self.disease_risk, 0.0, 1.0))

        # Update streaks for collapse condition
        for i in range(3):
            if self.grass[i] < 0.1:
                self.low_grass_streak[i] += 1
            else:
                self.low_grass_streak[i] = 0

        # Count only actual grazing actions (0,1,2) for balance metric
        if action in (0, 1, 2):
            self.grazing_counts[action] += 1
        self.prev_action = action

        # Check collapse (all low) AFTER updates
        if np.all(self.grass < 0.15):
            reward -= 60.0
            terminated = True
            truncated = False
        elif np.any(self.low_grass_streak > self.collapse_patience) or self.hunger >= 0.95 or self.thirst >= 0.95:
            # Pasture failed to recover
            reward -= 60.0
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = (self.day + 1) >= max(self.max_days, 45)

        self.day += 1

        obs = self._get_obs()
        info = self._get_info()

        if self.renderer:
            self.renderer.draw(
                self.grass, self.hunger, self.day,
                last_action=action, reward=reward,
                fert_levels=self.fertility, thirst=self.thirst,
                disease_risk=self.disease_risk, rain_flag=self.rain_flag
            )

        return obs, reward, terminated, truncated, info

    def render(self):  # Human render uses pygame; ansi returns text snapshot.
        if self.render_mode == "ansi":
            return (f"Day {self.day}\n"
                    f"Grass: A={self.grass[0]:.2f} B={self.grass[1]:.2f} C={self.grass[2]:.2f}\n"
                    f"Hunger={self.hunger:.2f}\n")
        # human mode already handled per step.
        return None

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None


def make_env(render_mode: Optional[str] = None, seed: Optional[int] = None, render_fps: int = 8) -> PastureEnv:
    return PastureEnv(render_mode=render_mode, seed=seed, render_fps=render_fps)


if __name__ == "__main__":  # Simple manual test
    env = PastureEnv(render_mode="ansi")
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        done = terminated or truncated
    print("Episode finished total_reward=", total_reward)
