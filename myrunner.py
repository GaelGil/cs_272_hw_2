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
    env.close()

    lambdas = [0, 0.3, 0.6, 0.9, 1.0]
    seeds = [0, 1, 2, 3, 4]
    threshold = 100
    table_rows = []
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
        first_reach_eps = []
        final_returns = []
        smoothed_runs = []
        for run in runs:
            smoothed = np.convolve(
                run,
                np.ones(window) / window,
                mode="valid",
            )
            smoothed_runs.append(smoothed)

            hits = np.where(smoothed >= threshold)[0]
            if len(hits) > 0:
                first_reach_eps.append(hits[0] + window - 1)

            final_returns.append(run[-window:].mean())

        smoothed_runs = np.array(smoothed_runs)
        mean_curve = smoothed_runs.mean(axis=0)
        spread = smoothed_runs.std(axis=0)
        episodes = np.arange(window - 1, len(runs[0]))

        mean_first_reach = (
            float(np.mean(first_reach_eps)) if first_reach_eps else None
        )
        mean_final_return = float(np.mean(final_returns))
        table_rows.append((lambda_, mean_first_reach, mean_final_return))

        plt.plot(episodes, mean_curve, label=f"Lambda={lambda_}")
        plt.fill_between(episodes, mean_curve - spread, mean_curve + spread, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Average return")
    plt.title("SARSA(lambda) Learning Curves Across Five Seeds")
    plt.legend()
    plt.savefig("lambda_sweep_plot")

    print(f"\nTarget return: {threshold} (100-episode moving average)")
    print(f"{'Lambda':<10} {'Episodes to target':<22} {'Mean final return':<20}")
    for lambda_, first_reach, final_return in table_rows:
        episodes_to_target = (
            "not reached" if first_reach is None else f"{first_reach:.1f}"
        )
        print(f"{lambda_:<10} {episodes_to_target:<22} {final_return:<20.2f}")
