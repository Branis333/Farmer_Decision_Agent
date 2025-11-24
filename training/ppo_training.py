"""PPO training with hyperparameter search for PastureEnv.

Usage:
    python training/ppo_training.py --timesteps 200000 --eval-episodes 5
"""
import argparse
import csv
import os
import sys
from itertools import product

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

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
    ap.add_argument('--timesteps', type=int, default=300000)
    ap.add_argument('--eval-episodes', type=int, default=5)
    ap.add_argument('--output', type=str, default='models/ppo')
    ap.add_argument('--runs', type=int, default=10)
    return ap.parse_args()


def build_grid():
    grid = {
        'learning_rate': [3e-4, 1e-3],
        'gamma': [0.95, 0.99],
        'n_steps': [128, 256],
        'batch_size': [64, 128],
        'ent_coef': [0.0, 0.02],
        'clip_range': [0.2, 0.3],
        'gae_lambda': [0.9, 0.95],
        'vf_coef': [0.5, 0.7],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]
    # Limit to a manageable set (choose first 20) but ensure >=10
    return combos[:20]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    combos = build_grid()
    if len(combos) < args.runs:
        # Duplicate with slight modifications
        while len(combos) < args.runs:
            mod = combos[-1].copy()
            mod['clip_range'] = 0.25 if mod['clip_range'] == 0.2 else 0.2
            combos.append(mod)

    keys = list(combos[0].keys())
    results_path = os.path.join(args.output, 'ppo_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward', 'std_reward', 'avg_length', 'grazing_balance_mean'])

    best_reward = float('-inf')
    best_model_path = None

    for i, params in enumerate(combos, start=1):
        print(f"[PPO] Run {i}/{len(combos)} params={params}")
        env_fn = lambda: make_env(render_mode=None)
        model = PPO(
            policy=MlpPolicy,
            env=env_fn(),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            ent_coef=params['ent_coef'],
            clip_range=params['clip_range'],
            gae_lambda=params['gae_lambda'],
            vf_coef=params['vf_coef'],
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps)
        metrics = evaluate_policy(env_fn, model, episodes=args.eval_episodes)
        print(f"Eval metrics: {metrics}")

        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i] + [params[k] for k in keys] + [metrics['avg_reward'], metrics['std_reward'], metrics['avg_length'], metrics['grazing_balance_mean']])

        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_model_path = os.path.join(args.output, 'best_ppo.zip')
            model.save(best_model_path)
            print(f"New best PPO saved at {best_model_path} avg_reward={best_reward:.2f}")
        model.env.close()

    print(f"Best PPO model: {best_model_path} reward={best_reward:.2f}")


if __name__ == '__main__':
    main()
