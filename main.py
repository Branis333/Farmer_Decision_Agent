"""Main runner for Pasture Allocation Optimizer.

Loads a trained model (DQN/PPO/A2C/REINFORCE) and runs a single
rendered evaluation episode to visualize agent behavior.

Usage (CMD):
	python main.py --algo dqn --render
	python main.py --algo ppo --render --episodes 2
	python main.py --algo reinforce --render

If --render omitted, uses ansi mode.
"""
import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

from environment.custom_env import PastureEnv, make_env

SB3_ALGOS = {
	'dqn': ('models/dqn/best_dqn.zip', 'stable_baselines3.DQN'),
	'ppo': ('models/ppo/best_ppo.zip', 'stable_baselines3.PPO'),
	'a2c': ('models/a2c/best_a2c.zip', 'stable_baselines3.A2C'),
}


def load_sb3(algo: str):
	from stable_baselines3 import DQN, PPO, A2C  # local import to avoid hard dependency if missing
	path, cls_name = SB3_ALGOS[algo]
	if not os.path.exists(path):
		raise FileNotFoundError(f"Model file not found: {path}. Train the {algo.upper()} model first.")
	if algo == 'dqn':
		return DQN.load(path)
	if algo == 'ppo':
		return PPO.load(path)
	if algo == 'a2c':
		return A2C.load(path)
	raise ValueError(algo)


class LoadedReinforcePolicy(torch.nn.Module):
	def __init__(self, hidden_size: int):
		super().__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(5, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, 3)
		)

	def forward(self, x):
		return self.net(x)

	def predict(self, obs):
		with torch.no_grad():
			logits = self.forward(torch.from_numpy(obs).float().unsqueeze(0))
			probs = torch.softmax(logits, dim=-1)
			dist = torch.distributions.Categorical(probs)
			return dist.sample().item()


def load_reinforce(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError(f"REINFORCE model not found at {path}. Train it first.")
	payload = torch.load(path, map_location='cpu')
	hidden_size = payload['config']['hidden_size']
	policy = LoadedReinforcePolicy(hidden_size)
	policy.load_state_dict(payload['state_dict'])
	policy.eval()
	return policy


def parse_args():
	ap = argparse.ArgumentParser()
	ap.add_argument('--algo', type=str, choices=['dqn', 'ppo', 'a2c', 'reinforce'], help='Algorithm to load')
	ap.add_argument('--episodes', type=int, default=1)
	ap.add_argument('--render', action='store_true')
	ap.add_argument('--delay', type=float, default=0.05, help='Delay between steps when ansi render mode used')
	ap.add_argument('--fps', type=int, default=6, help='Frames per second for human render (lower slows)')
	ap.add_argument('--hold', type=float, default=10.0, help='Seconds to keep window open after episode')
	ap.add_argument('--list', action='store_true', help='List available trained models and exit')
	return ap.parse_args()


def detect_available_models():
	available = {}
	for key, (path, _) in SB3_ALGOS.items():
		if os.path.exists(path):
			available[key] = path
	# Reinforce
	reinforce_path = 'models/reinforce/best_reinforce.pt'
	if os.path.exists(reinforce_path):
		available['reinforce'] = reinforce_path
	return available


def run_episode(model, render: bool, algo: str, delay: float, fps: int, hold: float):
	env = make_env(render_mode='human' if render else 'ansi', render_fps=fps)
	obs, info = env.reset()
	done = False
	total_reward = 0.0
	while not done:
		if hasattr(model, 'predict'):
			if algo in SB3_ALGOS:
				action, _ = model.predict(obs, deterministic=True)
			else:  # reinforce
				action = model.predict(obs)
		else:
			raise RuntimeError('Model has no predict method.')
		obs, r, terminated, truncated, info = env.step(int(action))
		total_reward += r
		done = terminated or truncated
		if not render:
			print(f"Day={info['day']} Grass={info['grass_levels']} Hunger={info['hunger']:.2f} Reward={r:.2f}")
			time.sleep(delay)
	# Hold window open for post-run viewing
	if render and hold > 0:
		end_t = time.time() + hold
		while time.time() < end_t:
			# Allow manual close
			for event in []:
				pass
			time.sleep(0.05)
	env.close()
	return total_reward


def main():
	args = parse_args()
	avail = detect_available_models()
	if args.list:
		print("Available models:")
		for algo, path in avail.items():
			print(f"  {algo}: {path}")
		print("Run: python main.py --algo <name> --render")
		return
	if not args.algo:
		if not avail:
			print("No trained model detected. Train one first (e.g. python training/dqn_training.py).")
			return
		# choose best based on modification time
		chosen = max(avail.items(), key=lambda kv: os.path.getmtime(kv[1]))[0]
		print(f"--algo not provided; auto-selecting most recent trained model: {chosen}")
		args.algo = chosen
	if args.algo in SB3_ALGOS:
		model = load_sb3(args.algo)
	else:
		model = load_reinforce('models/reinforce/best_reinforce.pt')
	rewards = []
	for ep in range(args.episodes):
		ep_reward = run_episode(model, render=args.render, algo=args.algo, delay=args.delay, fps=args.fps, hold=args.hold if ep == args.episodes - 1 else 0.0)
		rewards.append(ep_reward)
		print(f"Episode {ep+1}/{args.episodes} reward={ep_reward:.2f}")
	print(f"Average reward: {np.mean(rewards):.2f}")


if __name__ == '__main__':
	main()
