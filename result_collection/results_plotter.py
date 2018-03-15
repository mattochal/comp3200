import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from result_collection.helper_func import *

#
# def load_results(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     return data
#
#
# def get_policies(results):
#     policies = []
#     for experiment in results["results"]["seeds"]:
#         policies.append([experiment["P1"], experiment["P2"]])
#     return np.array(policies)
#
#
# def get_epoch_policies(results):
#     ps = []
#     for experiment in results["results"]["seeds"]:
#         p = []
#         for epoch in experiment["epoch"]:
#             p.append([epoch["P1"], epoch["P2"]])
#         ps.append(p)
#     return np.array(ps)


def plot_policies(results):
    fig, ax = plt.subplots()
    X = get_end_policies(results)
    colors = ["purple", "blue", "orange", "green", "red"]
    state = ["s0", "CC", "CD", "DC", "DD"]
    for s in range(5):
        ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=state[s])
    plt.title(results["config"]["simulation"]["agent_pair"] + " in " + results["config"]["simulation"]["game"])
    plt.xlabel('P(cooperation | state) for agent 0')
    plt.ylabel('P(cooperation | state) for agent 1')
    ax.legend(loc='best', shadow=True)
    plt.show()
    pass


def get_value_fns(results):
    vs = []
    for experiment in results["results"]["seeds"]:
        v = []
        for epoch in experiment["epoch"]:
            v.append([epoch["V1"], epoch["V2"]])
        vs.append(v)
    return np.array(vs)


def moving_average(a, window_size=10):
    ret = np.cumsum(a, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def plot_average_value(path):
    results = load_results(path)
    X = get_value_fns(results)

    avr_v1 = np.mean(X[:, :, 0], axis=0)
    min_v1 = avr_v1 - np.std(X[:, :, 0], axis=0)
    max_v1 = avr_v1 + np.std(X[:, :, 0], axis=0)

    avr_v2 = np.mean(X[:, :, 1], axis=0)
    min_v2 = avr_v2 - np.std(X[:, :, 1], axis=0)
    max_v2 = avr_v2 + np.std(X[:, :, 1], axis=0)

    avr_v1 = moving_average(avr_v1)
    min_v1 = moving_average(min_v1)
    max_v1 = moving_average(max_v1)

    avr_v2 = moving_average(avr_v2)
    min_v2 = moving_average(min_v2)
    max_v2 = moving_average(max_v2)

    x = np.arange(np.shape(min_v1)[0])
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, avr_v1, c="black")
    ax1.fill_between(x, min_v1, max_v1, color="blue", alpha=0.5)
    ax1.set_title('Value fn for agent 0')
    ax1.set_ylabel('Average reward per step')
    ax1.set_xlabel('Iterations')

    ax2.plot(x, avr_v2, c="black")
    ax2.fill_between(x, min_v2, max_v2, color="blue", alpha=0.5)
    ax2.set_title('Value fn for agent 1')
    ax2.set_ylabel('Average reward per step')
    ax2.set_xlabel('Iterations')
    plt.show()


def plot_connected_policy_dots(path):
    results = load_results(path)
    X = get_epoch_policies(results)[:10]
    # np.shape(X) = [repeats, epochs, # of players = 2, # of states = 5]

    fig, ax = plt.subplots()

    depth = 1000

    colors = ["purple", "blue", "orange", "green", "red"]
    state = ["s0", "CC", "CD", "DC", "DD"]

    for s in range(5):
        ax.scatter(X[:, depth-1, 0, s], X[:, depth-1, 1, s], s=25, c=colors[s], alpha=0.5, label=state[s])

    for repeat in X:
        for s in range(5):
            ax.plot(repeat[:depth, 0, s], repeat[:depth, 1, s], c=colors[s], alpha=0.5)

    plt.title(results["config"]["simulation"]["agent_pair"] + " in " + results["config"]["simulation"]["game"])
    plt.xlabel('P(cooperation | state) for agent 0')
    plt.ylabel('P(cooperation | state) for agent 1')
    ax.legend(loc='best', shadow=True)
    plt.show()


def plot_single_sim_run(path):
    results = load_results(path)
    plot_policies(results)


def plot_R_std_TFT_through_epochs(path):
    results = load_results(path)
    av_R1, std_R1, av_TFT1, _, av_R2, std_R2, av_TFT2, _ = get_av_epoch_R_std_TFT(results)
    X = [[av_R1, std_R1, av_TFT1], [av_R2, std_R2, av_TFT2]]

    fig, ax = plt.subplots()

    colors = ["b", "r"]
    labels = ["R", "TFT"]
    symbols = ["x", "+"]

    # 0 = R, 1 = TFT
    f, ax1 = plt.subplots()

    # # 0 = r, 1 = TFT
    # for rt in range(0, 2):
    #     # Two agents
    #     for a in range(2):
    #         R = X[a][rt]
    #
    #         avr_v1 = moving_average(R[0], window_size=1)
    #         min_v1 = moving_average(R[0]-R[1], window_size=1)
    #         max_v1 = moving_average(R[0]+R[1], window_size=1)
    #
    #         x = np.arange(np.shape(avr_v1)[0]) * 50
    #         ax1.plot(x, avr_v1, colors[rt]+symbols[a], alpha=0.5)
    #         ax1.fill_between(x, min_v1, max_v1, color=colors[rt], alpha=0.1)
    #
    #     ax1.set_ylabel(labels[rt])
    #     ax1.tick_params('y', colors=colors[rt])
    #     # break
    #     if rt == 0:
    #         ax1 = ax1.twinx()

    rt = 0
    # Two agents
    for a in range(2):
        R = X[a]

        avr_v1 = moving_average(R[0], window_size=1)
        min_v1 = moving_average(R[0] - R[1], window_size=1)
        max_v1 = moving_average(R[0] + R[1], window_size=1)

        x = np.arange(np.shape(avr_v1)[0]) * 50
        ax1.plot(x, avr_v1, colors[rt] + symbols[a], alpha=0.5, label="Agent " + str(a))
        ax1.fill_between(x, min_v1, max_v1, color=colors[rt], alpha=0.1)

    ax1.set_ylabel(labels[rt])
    ax1.tick_params('y', colors=colors[rt])

    ax2 = ax1.twinx()
    rt = 1
    for a in range(2):
        R = X[a]
        avr_v1 = moving_average(R[2], window_size=1)

        x = np.arange(np.shape(avr_v1)[0]) * 50
        ax2.plot(x, avr_v1, colors[rt] + symbols[a], alpha=0.5)

    ax2.set_ylabel(labels[rt])
    ax2.tick_params('y', colors=colors[rt])

    plt.title(results["config"]["simulation"]["agent_pair"] + " in " + results["config"]["simulation"]["game"])
    plt.xlabel('Iteration')

    ax1.legend(loc='lower right', shadow=True)
    plt.show()


if __name__ == "__main__":
    # plot_single_sim_run("results/lolaom_ST_space/S01xT08/result_lolaom_vs_lolaom_IPD.json")
    plot_R_std_TFT_through_epochs("results/lola1_random_init_policy_robustness/R25/lola1_vs_lola1_IPD.json")
    # plot_average_value("results/result_lolaom_vs_lolaom_IPD.json")
