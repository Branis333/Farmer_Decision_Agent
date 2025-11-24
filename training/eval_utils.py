from typing import Dict, Any
import inspect
import numpy as np


def evaluate_policy(env_fn, model, episodes: int = 5) -> Dict[str, Any]:
    """Run evaluation episodes and compute metrics.

    Supports both Stable-Baselines3 models (returning (action, state)) and
    custom policies returning a single action, with or without a
    'deterministic' parameter.
    """
    rewards = []
    lengths = []
    balances = []

    for _ in range(episodes):
        env = env_fn()
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done:
            action = None
            if hasattr(model, 'predict'):
                try:
                    sig = inspect.signature(model.predict)
                    if 'deterministic' in sig.parameters:
                        pred = model.predict(obs, deterministic=True)
                    else:
                        pred = model.predict(obs)
                except TypeError:
                    pred = model.predict(obs)
                if isinstance(pred, tuple):
                    action = pred[0]
                else:
                    action = pred
            else:
                # Fallback: direct act method if available
                if hasattr(model, 'act'):
                    action = model.act(obs)
                else:
                    raise RuntimeError('Model does not provide predict or act methods.')

            obs, r, terminated, truncated, info = env.step(int(action))
            ep_reward += r
            steps += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(steps)
        balances.append(info['grazing_balance'])
        env.close()

    rewards_arr = np.array(rewards)
    lengths_arr = np.array(lengths)
    balances_arr = np.array(balances)
    return {
        'avg_reward': float(rewards_arr.mean()),
        'std_reward': float(rewards_arr.std()),
        'avg_length': float(lengths_arr.mean()),
        'grazing_balance_mean': balances_arr.mean(axis=0).tolist(),
    }
