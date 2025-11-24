"""Custom REINFORCE (Monte Carlo Policy Gradient) training for PastureEnv.

Since Stable-Baselines3 does not provide a standalone REINFORCE implementation,
we implement a simple episodic policy gradient algorithm:
    - Collect full episode trajectories.
    - Compute discounted returns G_t.
    - Update policy via gradient of log π(a|s) * (G_t - baseline).
Includes entropy bonus and baseline as moving average of returns.

Usage:
    python training/reinforce_training.py --episodes 800 --eval-episodes 5
"""
import argparse
import csv
import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from training.eval_utils import evaluate_policy
    from environment.custom_env import make_env
except ModuleNotFoundError:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from training.eval_utils import evaluate_policy
    from environment.custom_env import make_env


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=1000)
    ap.add_argument('--eval-episodes', type=int, default=5)
    ap.add_argument('--output', type=str, default='models/reinforce')
    ap.add_argument('--runs', type=int, default=10)
    return ap.parse_args()


@dataclass
class ReinforceConfig:
    learning_rate: float
    gamma: float
    hidden_size: int
    entropy_coef: float


class ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

    def act(self, obs):
        with torch.no_grad():
            logits = self.forward(torch.from_numpy(obs).float().unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action

    def predict(self, obs):  # for eval_utils compatibility
        return self.act(obs)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def train_run(config: ReinforceConfig, episodes: int, eval_episodes: int, output_dir: str, run_id: int):
    env_fn = lambda: make_env(render_mode=None)
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = ReinforcePolicy(obs_dim, act_dim, config.hidden_size)
    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)

    baseline = None  # moving average of returns
    all_returns = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_rewards = []
        ep_log_probs = []
        ep_entropies = []
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            logits = policy.forward(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_entropies.append(entropy)
            obs = next_obs
            done = terminated or truncated

        returns = compute_returns(ep_rewards, config.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if baseline is None:
            baseline = returns_t.mean().item()
        # Update moving baseline
        baseline = 0.9 * baseline + 0.1 * returns_t.mean().item()
        adv = returns_t - baseline

        log_probs_t = torch.stack(ep_log_probs)
        entropies_t = torch.stack(ep_entropies)

        loss = -(log_probs_t * adv.detach()).mean() - config.entropy_coef * entropies_t.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ep_return = sum(ep_rewards)
        all_returns.append(ep_return)

        if ep % 50 == 0:
            avg_last = np.mean(all_returns[-50:])
            print(f"[REINFORCE Run {run_id}] Episode {ep}/{episodes} return={ep_return:.2f} avg_last50={avg_last:.2f}")

    env.close()
    # Evaluate trained policy
    metrics = evaluate_policy(env_fn, policy, episodes=eval_episodes)
    return policy, metrics


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Hyperparameter grid
    grid = {
        'learning_rate': [1e-3, 5e-4],
        'gamma': [0.95, 0.99],
        'hidden_size': [64, 128],
        'entropy_coef': [0.00, 0.01],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]
    if len(combos) < args.runs:
        while len(combos) < args.runs:
            mod = combos[-1].copy()
            mod['hidden_size'] = 96 if mod['hidden_size'] != 96 else 128
            combos.append(mod)

    results_path = os.path.join(args.output, 'reinforce_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward', 'std_reward', 'avg_length', 'grazing_balance_mean'])

    best_reward = float('-inf')
    best_model_path = None

    for i, params in enumerate(combos, start=1):
        cfg = ReinforceConfig(**params)
        print(f"[REINFORCE] Run {i}/{len(combos)} params={params}")
        policy, metrics = train_run(cfg, episodes=args.episodes, eval_episodes=args.eval_episodes, output_dir=args.output, run_id=i)
        print(f"Eval metrics: {metrics}")

        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i] + [params[k] for k in keys] + [metrics['avg_reward'], metrics['std_reward'], metrics['avg_length'], metrics['grazing_balance_mean']])

        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_model_path = os.path.join(args.output, 'best_reinforce.pt')
            torch.save({'state_dict': policy.state_dict(), 'config': params}, best_model_path)
            print(f"New best REINFORCE saved at {best_model_path} avg_reward={best_reward:.2f}")

    print(f"Best REINFORCE model: {best_model_path} reward={best_reward:.2f}")


if __name__ == '__main__':
    main()
