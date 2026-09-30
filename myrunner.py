import importlib

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from myagent import SarsaLambdaAgent

# my editor removes import myenv since its never used
importlib.import_module("myenv")


if __name__ == "__main__":
    env = gym.make("cs272/MyEnv-v0", render_mode="ansi")
    agent = SarsaLambdaAgent(env, lam=0.1, seed=0, total_epi=5000)
    agent.learn()
    episode, finished = agent.best_run(render=True)
    print("Finished", finished)
    print("Return", agent.calc_return(episode=episode))

    lambdas = [0, 0.3, 0.6, 0.9, 1.0]
    seeds = [0, 1, 2, 3, 4]
    window = 100
    results = {}

    # train an agent with the given lambdas each with seed of 0-4.
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
        smoothed_runs = np.array(
            [np.convolve(run, np.ones(window) / window, mode="valid") for run in runs]
        )

        mean_curve = smoothed_runs.mean(axis=0)
        spread = smoothed_runs.std(axis=0)
        episodes = np.arange(window - 1, len(runs[0]))

        plt.plot(episodes, mean_curve, label=f"Lambda={lambda_}")
        plt.fill_between(episodes, mean_curve - spread, mean_curve + spread, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Average return")
    plt.legend()
    plt.savefig("lambda_learning_curves")
