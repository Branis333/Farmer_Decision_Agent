"""A2C training with hyperparameter search for PastureEnv.

Usage:
    python training/a2c_training.py --timesteps 200000 --eval-episodes 5
"""
import argparse
import csv
import os
import sys
from itertools import product

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy

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
    ap.add_argument('--output', type=str, default='models/a2c')
    ap.add_argument('--runs', type=int, default=10)
    return ap.parse_args()


def build_grid():
    grid = {
        'learning_rate': [7e-4, 1e-3],
        'gamma': [0.95, 0.99],
        'n_steps': [5, 10, 20],
        'ent_coef': [0.0, 0.01],
        'gae_lambda': [0.95, 0.9],
        'vf_coef': [0.5, 0.7],
    }
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]
    return combos[:25]


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    combos = build_grid()
    if len(combos) < args.runs:
        while len(combos) < args.runs:
            mod = combos[-1].copy()
            mod['n_steps'] = 15 if mod['n_steps'] != 15 else 5
            combos.append(mod)
    keys = list(combos[0].keys())

    results_path = os.path.join(args.output, 'a2c_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward', 'std_reward', 'avg_length', 'grazing_balance_mean'])

    best_reward = float('-inf')
    best_model_path = None

    for i, params in enumerate(combos, start=1):
        print(f"[A2C] Run {i}/{len(combos)} params={params}")
        env_fn = lambda: make_env(render_mode=None)
        model = A2C(
            policy=MlpPolicy,
            env=env_fn(),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            n_steps=params['n_steps'],
            ent_coef=params['ent_coef'],
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
            best_model_path = os.path.join(args.output, 'best_a2c.zip')
            model.save(best_model_path)
            print(f"New best A2C saved at {best_model_path} avg_reward={best_reward:.2f}")
        model.env.close()

    print(f"Best A2C model: {best_model_path} reward={best_reward:.2f}")


if __name__ == '__main__':
    main()
