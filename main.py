
import argparse
import os
import time
from typing import Optional

import torch

from environment.custom_env import PastureEnv, make_env

SB3_ALGOS = {
	'dqn': ('models/dqn/best_dqn.zip', 'stable_baselines3.DQN'),
	'ppo': ('models/ppo/best_ppo.zip', 'stable_baselines3.PPO'),
	'a2c': ('models/a2c/best_a2c.zip', 'stable_baselines3.A2C'),
}


def load_sb3(algo: str):
	"""Load SB3 model and wrap it for NumPy 2.x compatibility."""
	from stable_baselines3 import DQN, PPO, A2C
	path, cls_name = SB3_ALGOS[algo]
	if not os.path.exists(path):
		raise FileNotFoundError(f"Model file not found: {path}. Train the {algo.upper()} model first.")
	if algo == 'dqn':
		model = DQN.load(path)
	elif algo == 'ppo':
		model = PPO.load(path)
	elif algo == 'a2c':
		model = A2C.load(path)
	else:
		raise ValueError(algo)
	return SB3ModelWrapper(model)


class SB3ModelWrapper:
	"""Wrapper to bypass NumPy 2.x compatibility issues with SB3 predict."""
	def __init__(self, model):
		self.model = model
		self.policy = model.policy
		self.device = model.device
	
	def predict(self, obs, deterministic=True):
		
		obs_list = obs.tolist() if hasattr(obs, 'tolist') else list(obs)
		obs_tensor = torch.tensor(obs_list, dtype=torch.float32, device=self.device)
		obs_tensor = obs_tensor.unsqueeze(0)  
		with torch.no_grad():
			action = self.policy._predict(obs_tensor, deterministic=deterministic)
			return int(action.cpu().item()), None


class LoadedReinforcePolicy(torch.nn.Module):
	def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
		super().__init__()
		self.net = torch.nn.Sequential(
			torch.nn.Linear(obs_dim, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_size, act_dim)
		)

	def forward(self, x):
		return self.net(x)

	def predict(self, obs):
		with torch.no_grad():
			
			obs_list = obs.tolist() if hasattr(obs, 'tolist') else list(obs)
			obs_tensor = torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0)
			logits = self.forward(obs_tensor)
			probs = torch.softmax(logits, dim=-1)
			dist = torch.distributions.Categorical(probs)
			return dist.sample().item()


def load_reinforce(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError(f"REINFORCE model not found at {path}. Train it first.")
	payload = torch.load(path, map_location='cpu')
	state = payload['state_dict']
	
	first_w = state['net.0.weight'] 
	last_w = state['net.4.weight']   
	hidden_size = first_w.shape[0]
	obs_dim = first_w.shape[1]
	act_dim = last_w.shape[0]
	policy = LoadedReinforcePolicy(obs_dim, act_dim, hidden_size)
	policy.load_state_dict(state)
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
			grass = info['grass_levels']
			grass_str = f"[{grass[0]:.2f}, {grass[1]:.2f}, {grass[2]:.2f}]"
			print(f"Day={info['day']} Grass={grass_str} Hunger={info['hunger']:.2f} Reward={r:.2f}")
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
	avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
	print(f"Average reward: {avg_reward:.2f}")


if __name__ == '__main__':
	main()
