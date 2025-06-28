"""Training script for the top-down racing environment.

Usage:
    python train.py --algorithm ppo --total-timesteps 50000 --n-envs 4
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env import RacingEnv


def make_env(render: bool = False):
    def _init():
        env = RacingEnv(render_mode="human" if render else None)
        return Monitor(env)

    return _init


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--out", type=str, default="models/racing_agent.zip")
    return parser.parse_args()


def main():
    args = parse_args()
    env_fns = [make_env() for _ in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    if args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
        )
    else:
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            verbose=1,
        )

    model.learn(total_timesteps=args.total_timesteps)
    Path("models").mkdir(exist_ok=True)
    model.save(args.out)


if __name__ == "__main__":
    main()
