"""Plot training rewards stored by the Monitor wrapper."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(log_path: str):
    df = pd.read_csv(log_path)
    plt.plot(df["l"])  # episode reward
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py path/to/monitor.csv")
        sys.exit(1)
    main(sys.argv[1])
