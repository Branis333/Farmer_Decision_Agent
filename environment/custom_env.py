import math
import random
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from .rendering import PastureRenderer
except Exception:  # pragma: no cover - allow direct script usage
    PastureRenderer = None


class PastureEnv(gym.Env):
    """Custom rotational grazing environment for a Farmer Decision Agent.

    State vector (observation):
        [grass_A, grass_B, grass_C, cow_hunger, day_normalized]
        All values in [0,1]. Day normalized = current_day / max_days.

    Action space (Discrete 3):
        0 -> Graze Pasture A
        1 -> Graze Pasture B
        2 -> Graze Pasture C

    Dynamics:
        - Grazed pasture: grass -= depletion_rate, hunger -= feeding_amount
        - Resting pasture: grass += regrowth_rate
        - Hunger increases slightly each step (base_hunger_increase) then is reduced if grazed.
        - Values are clipped to [0,1].

    Reward shaping:
        +20 if chosen pasture grass >= 0.5 (healthy grazing)
        -15 if chosen pasture grass < 0.3 before grazing (overgrazed usage)
        +10 if hunger reduced (i.e., previous hunger - new hunger > 0)
        -50 if all pastures < 0.2 after update => system collapse (episode terminates)

    Termination conditions:
        - System collapse (all grass levels < 0.2) => truncated = False, terminated = True
        - Any pasture stays below 0.1 for > collapse_patience consecutive days (terminated)
        - Max days reached (truncated)

    Additional info metrics returned in 'info':
        grass_levels, hunger, grazing_balance (counts of each pasture chosen), day
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 5}

    def __init__(self,
                 max_days: int = 30,
                 depletion_rate: float = 0.3,
                 feeding_amount: float = 0.4,
                 regrowth_rate: float = 0.1,
                 base_hunger_increase: float = 0.05,
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
        self.collapse_patience = collapse_patience
        self.render_mode = render_mode
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Discrete actions: choose one pasture.
        self.action_space = spaces.Discrete(3)
        # Observation: 5 continuous features in [0,1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # Internal state
        self.grass = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.hunger: float = 0.5
        self.day: int = 0
        self.low_grass_streak = np.zeros(3, dtype=np.int32)  # consecutive days < 0.1 per pasture
        self.grazing_counts = np.zeros(3, dtype=np.int32)

        self.renderer = None
        if self.render_mode == "human" and PastureRenderer is not None:
            self.renderer = PastureRenderer(window_width=600, window_height=400, max_fps=render_fps)

    def seed(self, seed: Optional[int] = None):  # Gymnasium seeding compatibility
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _get_obs(self) -> np.ndarray:
        day_norm = self.day / self.max_days
        return np.array([
            self.grass[0],
            self.grass[1],
            self.grass[2],
            self.hunger,
            day_norm
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        grazing_balance = (self.grazing_counts / max(1, np.sum(self.grazing_counts))).tolist()
        return {
            "grass_levels": self.grass.copy(),
            "hunger": self.hunger,
            "day": self.day,
            "grazing_balance": grazing_balance,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self.grass = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        self.hunger = 0.5
        self.day = 0
        self.low_grass_streak[:] = 0
        self.grazing_counts[:] = 0
        obs = self._get_obs()
        info = self._get_info()
        if self.renderer:
            self.renderer.reset()
            self.renderer.draw(self.grass, self.hunger, self.day, last_action=None, reward=0.0)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        prev_hunger = self.hunger

        # Apply grazing to chosen pasture
        chosen_grass_before = float(self.grass[action])
        reward = 0.0

        # Reward component: healthy vs overgrazed BEFORE depletion
        if chosen_grass_before >= 0.5:
            reward += 20.0
        elif chosen_grass_before < 0.3:
            reward -= 15.0

        # Grazing effect
        self.grass[action] -= self.depletion_rate
        self.hunger -= self.feeding_amount

        # Resting pastures regrow
        for i in range(3):
            if i != action:
                self.grass[i] += self.regrowth_rate

        # Base hunger natural increase (cows get hungry over time)
        self.hunger += self.base_hunger_increase

        # Clip values
        self.grass = np.clip(self.grass, 0.0, 1.0)
        self.hunger = float(np.clip(self.hunger, 0.0, 1.0))

        # Hunger improvement reward
        if prev_hunger - self.hunger > 0:  # hunger decreased
            reward += 10.0

        # Update streaks for collapse condition
        for i in range(3):
            if self.grass[i] < 0.1:
                self.low_grass_streak[i] += 1
            else:
                self.low_grass_streak[i] = 0

        # Count grazing for balance metric
        self.grazing_counts[action] += 1

        # Check collapse (all low) AFTER updates
        if np.all(self.grass < 0.2):
            reward -= 50.0
            terminated = True
            truncated = False
        elif np.any(self.low_grass_streak > self.collapse_patience):
            # Pasture failed to recover
            reward -= 50.0
            terminated = True
            truncated = False
        else:
            terminated = False
            truncated = (self.day + 1) >= self.max_days

        self.day += 1

        obs = self._get_obs()
        info = self._get_info()

        if self.renderer:
            self.renderer.draw(self.grass, self.hunger, self.day, last_action=action, reward=reward)

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
