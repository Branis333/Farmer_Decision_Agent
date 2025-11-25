# Farmer Decision Agent 🌾🐄

A reinforcement learning project that trains agents to manage cattle grazing across multiple pastures. The agent must balance grass consumption, animal hunger/thirst, and pasture sustainability over a 30-day simulation.

## Project Overview

The agent controls a farmer deciding which pasture to graze cattle on each day. The environment simulates:
- **3 Pastures** with grass levels and fertility
- **Cattle needs**: hunger and thirst management
- **Weather events**: rain affects grass regrowth
- **Disease risk**: poor management increases health risks

### Action Space
| Action | Description |
|--------|-------------|
| 0 | Graze on Pasture 1 |
| 1 | Graze on Pasture 2 |
| 2 | Graze on Pasture 3 |
| 3 | Rest (don't graze) |
| 4 | Go to water source |

### Observation Space (11 features)
- Grass levels (3 pastures)
- Fertility levels (3 pastures)
- Hunger level
- Thirst level
- Disease risk
- Rain flag
- Day progress (normalized)

---

## Project Structure

```
Farmer_Decision_Agent/
├── main.py              # Run trained models
├── random_demo.py       # Baseline: random actions
├── requirements.txt     # Dependencies
├── environment/
│   ├── custom_env.py    # PastureEnv gymnasium environment
│   └── rendering.py     # Pygame visualization
├── models/              # Saved trained models
│   ├── dqn/best_dqn.zip
│   ├── ppo/best_ppo.zip
│   ├── a2c/best_a2c.zip
│   └── reinforce/best_reinforce.pt
└── training/            # Training scripts & notebooks
    ├── dqn_training.py
    ├── ppo_training.ipynb
    ├── a2c_training.ipynb
    ├── reinforce_training.ipynb
    └── plot.ipynb       # Results visualization
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Branis333/Farmer_Decision_Agent.git
cd Farmer_Decision_Agent

# Create conda environment (recommended)
conda create -n rl_env python=3.11 -y
conda activate rl_env

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.11+
- PyTorch
- Stable-Baselines3
- Gymnasium
- Pygame (for visualization)

---

## Quick Start

### 1. Check Available Models
```bash
python main.py --list
```

### 2. Run a Trained Model
```bash
# Run with pygame visualization
python main.py --algo dqn --render --episodes 5

# Run without visualization (text output)
python main.py --algo ppo --episodes 10
```

### 3. Compare with Random Baseline
```bash
# Random policy with visualization
python random_demo.py --episodes 3

# Random policy (text only)
python random_demo.py --episodes 5 --no-render
```

---

## Running Trained Models

### Command: `main.py`

```bash
python main.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--algo` | Algorithm: `dqn`, `ppo`, `a2c`, `reinforce` | Auto-detect |
| `--episodes` | Number of episodes to run | 1 |
| `--render` | Enable pygame visualization | False |
| `--fps` | Frames per second (lower = slower) | 6 |
| `--hold` | Seconds to keep window open after episode | 10.0 |
| `--delay` | Delay between steps (text mode) | 0.05 |
| `--list` | List available trained models | - |

### Examples

#### DQN (Deep Q-Network)
```bash
# Visual demo - 5 episodes
python main.py --algo dqn --render --episodes 5

# Benchmark - 50 episodes (no render for speed)
python main.py --algo dqn --episodes 50
```

#### PPO (Proximal Policy Optimization)
```bash
# Visual demo
python main.py --algo ppo --render --episodes 3

# Slow visualization for presentation
python main.py --algo ppo --render --fps 3 --episodes 1
```

#### A2C (Advantage Actor-Critic)
```bash
# Visual demo
python main.py --algo a2c --render --episodes 5

# Extended evaluation
python main.py --algo a2c --episodes 100
```

#### REINFORCE (Policy Gradient)
```bash
# Visual demo
python main.py --algo reinforce --render --episodes 5

# Quick test
python main.py --algo reinforce --episodes 10
```

---

## Random Baseline Demo

### Command: `random_demo.py`

Shows why intelligent policies are needed by running random actions.

```bash
python random_demo.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes` | Number of episodes | 1 |
| `--no-render` | Disable pygame (text only) | False |
| `--fps` | Frames per second | 8 |
| `--hold` | Window hold time after episode | 5.0 |

### Examples

```bash
# Watch random agent fail (pygame)
python random_demo.py --episodes 3

# Text output for logging
python random_demo.py --episodes 10 --no-render

# Slow visualization
python random_demo.py --fps 4 --episodes 2
```

---

## Training New Models

Training scripts are in the `training/` folder:

```bash
# DQN (Python script)
python training/dqn_training.py

# PPO, A2C, REINFORCE (Jupyter notebooks)
jupyter notebook training/ppo_training.ipynb
```

Models are automatically saved to the `models/` directory.

---

## Visualization & Analysis

Generate comparison plots using the plotting notebook:

```bash
jupyter notebook training/plot.ipynb
```

This evaluates all trained models and creates:
- Reward distribution box plots
- Average performance bar charts
- Multi-metric comparisons
- Episode-by-episode performance lines

---

## Expected Results

| Algorithm | Avg Reward | Stability |
|-----------|------------|-----------|
| DQN | ~80-120 | Good |
| PPO | ~90-130 | Excellent |
| A2C | ~70-110 | Moderate |
| REINFORCE | ~60-100 | Variable |

*Results vary based on training hyperparameters and random seeds.*

---

## Troubleshooting

### NumPy 2.x Compatibility Error
If you see `RuntimeError: Could not infer dtype of numpy.float32`:
```bash
pip install "numpy<2"
```

### Pygame Window Not Opening
Ensure pygame is installed:
```bash
pip install pygame
```

### Model Not Found
Train the model first or check the `models/` directory:
```bash
python main.py --list
```

---

## Author

Branis333
