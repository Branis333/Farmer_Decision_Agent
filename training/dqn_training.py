"""DQN training with hyperparameter search for PastureEnv.

Conducts multiple runs (>=10) over a parameter grid. Saves best model
based on average evaluation reward. Logs results to CSV.

Usage:
    python training/dqn_training.py --episodes 30000 --eval-episodes 5
"""
import argparse
import csv
import os
import sys
from itertools import product

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

try:
    from training.eval_utils import evaluate_policy
    from environment.custom_env import make_env
except ModuleNotFoundError:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from training.eval_utils import evaluate_policy
    from environment.custom_env import make_env


def make_dqn(env_fn, params):
    return DQN(
        policy=MlpPolicy,
        env=env_fn(),
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        batch_size=params['batch_size'],
        buffer_size=params['buffer_size'],
        train_freq=params['train_freq'],
        target_update_interval=params['target_update_interval'],
        exploration_fraction=params['exploration_fraction'],
        exploration_initial_eps=params['exploration_initial_eps'],
        exploration_final_eps=params['exploration_final_eps'],
        verbose=0,
        tensorboard_log=params.get('tb_log', None),
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=50000, help='Total timesteps to train per run')
    ap.add_argument('--eval-episodes', type=int, default=5)
    ap.add_argument('--output', type=str, default='models/dqn')
    ap.add_argument('--runs', type=int, default=10, help='Minimum number of runs')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Hyperparameter grid (ensure >=10 combinations)
    grid = {
        'learning_rate': [1e-3, 5e-4],
        'gamma': [0.95, 0.99],
        'batch_size': [32, 64],
        'buffer_size': [5000, 20000],
        'train_freq': [4],
        'target_update_interval': [1000, 2000],
        'exploration_fraction': [0.2, 0.4],
        'exploration_initial_eps': [1.0],
        'exploration_final_eps': [0.05],
    }
    # Build combinations
    keys = list(grid.keys())
    combos = []
    for values in product(*[grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        combo['tb_log'] = None
        combos.append(combo)
    # Guarantee at least args.runs
    if len(combos) < args.runs:
        # Duplicate some combos with slight modifications (e.g., gamma tweak)
        while len(combos) < args.runs:
            mod = combos[-1].copy()
            mod['gamma'] = 0.9 if mod['gamma'] != 0.9 else 0.99
            combos.append(mod)

    results_path = os.path.join(args.output, 'dqn_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_id'] + keys + ['avg_reward', 'std_reward', 'avg_length', 'grazing_balance_mean'])

    best_reward = float('-inf')
    best_model_path = None
    run_id = 0

    for params in combos:
        run_id += 1
        print(f"[DQN] Run {run_id}/{len(combos)} params={params}")
        env_fn = lambda: make_env(render_mode=None)
        model = make_dqn(env_fn, params)
        model.learn(total_timesteps=args.episodes)
        metrics = evaluate_policy(env_fn, model, episodes=args.eval_episodes)
        print(f"Eval metrics: {metrics}")

        with open(results_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([run_id] + [params[k] for k in keys] + [metrics['avg_reward'], metrics['std_reward'], metrics['avg_length'], metrics['grazing_balance_mean']])

        if metrics['avg_reward'] > best_reward:
            best_reward = metrics['avg_reward']
            best_model_path = os.path.join(args.output, f"best_dqn.zip")
            model.save(best_model_path)
            print(f"New best model saved at {best_model_path} avg_reward={best_reward:.2f}")
        model.env.close()

    print(f"Best DQN model: {best_model_path} with avg_reward={best_reward:.2f}")


if __name__ == '__main__':
    main()
