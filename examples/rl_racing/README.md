# Reinforcement Learning Racing Example

This example demonstrates a simple top-down racing environment and training setup.

## Files

- `env.py` – environment implementation based on `gymnasium` and `pygame`.
- `train.py` – training script using Stable Baselines3 (PPO or DQN).
- `visualize.py` – helper script to plot episode rewards from the training log.
- `models/` – directory where trained models are saved.

## Usage

Install dependencies (e.g. `gymnasium`, `stable-baselines3`, `pygame`).

```bash
pip install gymnasium stable-baselines3 pygame matplotlib pandas
```

Train an agent:

```bash
python train.py --algorithm ppo --total-timesteps 50000 --n-envs 4
```

Visualize training rewards:

```bash
python visualize.py monitor.csv
```
