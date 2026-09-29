import gymnasium as gym
import numpy as np

from myagent import SarsaLambdaAgent, argmax_action

if __name__ == "__main__":
    lambdas = [0, 0.3, 0.6, 0.9, 1.0]
    seeds = [0, 1, 2, 3, 4]
    results = {}

    for lambda_ in lambdas:
        runs = []
        for seed in seeds:

        env = gym.make("cs272/MyEnv-v0")
        agent = SarsaLambdaAgent(env, lam=lambda_, seed=42, total_epi=5000)
        returns = agent.learn()
        runs.append(returns)


    results[lambda_] = np.array(runs)
    rng = np.random.default_rng(42)

    print(agent.q.shape)
    print(agent.q[0])

    print(f"argmax_action: {argmax_action(np.array([1.0, 5.0, 10.0]), rng)}")

    action_count = [0, 0, 0]
    for i in range(1000):
        action = argmax_action(np.array([1.0, 1.0, 1.0]), rng)
        action_count[action] += 1
    print(action_count)

    explore_count = 0
    agent.q[10] = [0.0, 0.0, 10.0]
    for i in range(1000):
        if agent.eps_greedy(10) != 2:
            explore_count += 1
    print(explore_count)

    agent2 = SarsaLambdaAgent(env, lam=0.9, total_epi=2000, seed=42)
    returns = agent2.learn()
    print(sum(returns[:100]) / 100, sum(returns[-100:]) / 100)

    episode = [(0, 1, -1.0), (4, 1, 40.0), (0, 2, -1.0), (3, 2, -20.0)]
    print(agent2.calc_return(episode))
    print(agent2.calc_return(episode, discounted=True))
