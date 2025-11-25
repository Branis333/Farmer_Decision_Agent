"""Random action demonstration for PastureEnv.

Shows why an intelligent policy is needed. Runs one or multiple episodes
with random actions while rendering via pygame.

Usage (CMD):
    python random_demo.py --episodes 2 --no-render   # headless ansi
    python random_demo.py --episodes 1               # pygame window
"""
import argparse
import time
from statistics import mean

from environment.custom_env import PastureEnv, make_env


def run_random(episodes: int, render: bool, fps: int, hold: float):
    rewards = []
    for ep in range(episodes):
        env = make_env(render_mode="human" if render else "ansi", render_fps=fps)
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            done = terminated or truncated
            if not render:
               
                print(f"Step day={info['day']} grass={info['grass_levels']} fert={info.get('fertility')} hunger={info['hunger']:.2f} thirst={info.get('thirst',0):.2f} reward={r:.2f}")
                time.sleep(0.05)
        rewards.append(ep_reward)
        print(f"Episode {ep+1}/{episodes} random policy total reward={ep_reward:.2f}")
        if render and hold > 0:
            
            t_end = time.time() + hold
            while time.time() < t_end:
               
                for event in env.renderer and [] or []:
                    pass
                time.sleep(0.05)
        env.close()
    print(f"Average reward over {episodes} random episodes: {mean(rewards):.2f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--no-render", action="store_true", help="Use ANSI text instead of pygame window")
    ap.add_argument("--fps", type=int, default=4, help="Frames per second for human rendering (lower = slower)")
    ap.add_argument("--hold", type=float, default=5.0, help="Seconds to hold window after episode")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_random(args.episodes, render=not args.no_render, fps=args.fps, hold=args.hold)
