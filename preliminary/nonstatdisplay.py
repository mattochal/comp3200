import matplotlib.pyplot as plt
import numpy as np

from preliminary import q_learner_vs_tft_ipd
from preliminary import q_learner_vs_q_learner


def moving_average(a, n=30):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

if __name__ == "__main__":
    n = 10000
    a1, a2, = q_learner_vs_q_learner.basic_game(n, g=0.90, print_summary=True)

    data = np.zeros(n)
    for i, t in enumerate(a1.transitions):
        (state, action, reward, new_state) = t
        data[i] = reward

    plt.plot(moving_average(data), label="Q-learner vs Q-learner")

    a1, a2, = q_learner_vs_tft_ipd.basic_game(n, g=0.90, print_summary=True)
    data = np.zeros(n)
    for i, t in enumerate(a1.transitions):
        (state, action, reward, new_state) = t
        data[i] = reward

    # plt.figure(figsize=(8, 5))2

    plt.plot(moving_average(data), label="Q-learner vs TFT-agent")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    # labels
    plt.xlabel("Iterations")
    plt.ylabel("Average reward per step")

    plt.show()