import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats

from preliminary import q_learner_vs_tft_ipd
from preliminary import q_learner_vs_q_learner
from preliminary import pgapp_vs_pgapp, pgapp_vs_tft


def moving_average(a, n=30):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


if __name__ == "__main__":
    n = 10000
    repeats = 30
    window = 10

    try:
        data = np.load("results/qvq.npy")
    except:
        a = []
        for r in range(repeats):
            a1, a2, = q_learner_vs_q_learner.basic_game(n, g=0.90, print_summary=True)
            a.append(a1)
        data = np.zeros((repeats, n))

        for r in range(repeats):
            for i, t in enumerate(a[r].transitions):
                (state, action, reward, new_state) = t
                data[r][i] = reward
        np.save('results/qvq.npy', data)

    h = mean_confidence_interval(data)
    data_av = np.mean(data, axis=0)
    x = np.arange(data_av.shape[0]-window+1)
    plt.plot(moving_average(data_av, window), label="Q-learner vs Q-learner", alpha=0.4, c="red")
    plt.fill_between(x, moving_average(data_av-h, window), moving_average(data_av+h, window), alpha=0.1, color="red")

    # Q vs TFT
    try:
        data = np.load("results/qvtft.npy")
    except:
        a = []
        for r in range(repeats):
            a1, a2, = q_learner_vs_tft_ipd.basic_game(n, g=0.90, print_summary=True)
            a.append(a1)
        data = np.zeros((repeats, n))

        for r in range(repeats):
            for i, t in enumerate(a[r].transitions):
                (state, action, reward, new_state) = t
                data[r][i] = reward
        np.save('results/qvtft.npy', data)

    h = mean_confidence_interval(data)
    data_av = np.mean(data, axis=0)
    x = np.arange(data_av.shape[0]-window+1)
    plt.plot(moving_average(data_av, window), label="Q-learner vs TFT-agent", alpha=0.4, c="green")
    plt.fill_between(x, moving_average(data_av-h, window), moving_average(data_av+h, window), alpha=0.15, color="green")


    # PGA PP vs PGA-PP
    try:
        # raise Exception()
        data = np.load("results/pgavpga.npy")
    except:
        a = []
        for r in range(repeats):
            a1, a2, = pgapp_vs_pgapp.basic_game(n, g=0.90, print_summary=True)
            a.append(a1)
        data = np.zeros((repeats, n))

        for r in range(repeats):
            for i, t in enumerate(a[r].transitions):
                (state, action, reward, new_state) = t
                data[r][i] = reward
        np.save('results/pgavpga.npy', data)

    h = mean_confidence_interval(data)
    data_av = np.mean(data, axis=0)
    x = np.arange(data_av.shape[0]-window+1)
    plt.plot(moving_average(data_av, window), label="PGA-PP vs PGA-PP", alpha=0.5, c="orange")
    plt.fill_between(x, moving_average(data_av-h, window), moving_average(data_av+h, window), alpha=0.2, color="orange")


    # PGA PP vs PGA-PP
    try:
        # raise Exception()
        data = np.load("results/pgavtft.npy")
    except:
        a = []
        for r in range(repeats):
            a1, a2, = pgapp_vs_tft.basic_game(n, g=0.90, print_summary=True)
            a.append(a1)
        data = np.zeros((repeats, n))

        for r in range(repeats):
            for i, t in enumerate(a[r].transitions):
                (state, action, reward, new_state) = t
                data[r][i] = reward
        np.save('results/pgavtft.npy', data)

    h = mean_confidence_interval(data)
    data_av = np.mean(data, axis=0)
    x = np.arange(data_av.shape[0] - window + 1)
    plt.plot(moving_average(data_av, window), label="PGA-PP vs TFT-agent", alpha=0.4, c="blue")
    plt.fill_between(x, moving_average(data_av - h, window), moving_average(data_av + h, window), alpha=0.1, color="blue")

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

    # labels
    plt.xlabel("Iterations")
    plt.ylabel("Average reward per step")

    plt.savefig("q-learner-nonstationarity-graph.pdf")