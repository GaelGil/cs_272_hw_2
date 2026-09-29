import importlib

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from myagent import SarsaLambdaAgent

# my editor removes import myenv since its never used
importlib.import_module("myenv")


if __name__ == "__main__":
    lambdas = [0, 0.3, 0.6, 0.9, 1.0]
    seeds = [0, 1, 2, 3, 4]
    results = {}

    for lambda_ in lambdas:
        runs = []
        for seed in seeds:
            env = gym.make("cs272/MyEnv-v0")
            agent = SarsaLambdaAgent(env, lam=lambda_, seed=seed, total_epi=5000)
            returns = agent.learn()
            runs.append(returns)
        env.close()
        results[lambda_] = np.array(runs)

    for lambda_, runs in results.items():
        average_curve = runs.mean(axis=0)
        plt.plot(average_curve, label=f"Lambda={lambda_}")

    plt.xlabel("Episode")
    plt.ylabel("Average return")
    plt.legend()
    plt.savefig("lambda_learning_curves")
