import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from result_collection.helper_func import *


def plot_policies(results, states=["s0", "CC", "CD", "DC", "DD"], title=""):
    fig, ax = plt.subplots()
    X = get_end_policies(results)
    X = np.array(X)
    colors = ["purple", "blue", "orange", "green", "red"]
    for s in range(5):
        ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=states[s])
    plt.title(title)
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


def get_value_fns_from_policy(results):
    r1 = np.array(results["config"]["game"]["payoff1"])
    r2 = np.array(results["config"]["game"]["payoff2"])
    gamma = results["config"]["agent_pair"]["gamma"]
    vs = []
    for experiment in results["results"]["seeds"]:
        v = []
        for epoch in experiment["epoch"]:
            V1, V2 = calculate_value_fn_from_policy(np.array([epoch["V1"], epoch["V2"]]), r1, r2, gamma)
            v.append([V1, V2])
        vs.append(v)
    return np.array(vs)


def moving_average(a, window_size=10):
    ret = np.cumsum(a, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def plot_average_value(results):
    # results = load_results(path)
    X = get_value_fns_from_policy(results)

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


def set_legend_ax(ax, ncols=2):
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), shadow=True, ncol=ncols)


def plot_policies_and_v_timeline(ordered_results, states=["s0", "CC", "CD", "DC", "DD"], title="", filename_to_save="",
                                 prob_state="cooperation"):
    cols = len(ordered_results) + 1
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(14, 5))
    plt.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.08, wspace=0.22, hspace=0.27)

    colors = ["cyan", "blue", "orange", "green", "red"]
    # for r, row in enumerate(axes):
    for c, ax in enumerate(axes):
        X = get_end_policies(ordered_results[c])
        agent_pair = ordered_results[c]["config"]["simulation"]["agent_pair"]
        pair = viewer_friendly_pair(agent_pair, as_list=True)
        X = np.array(X)
        for s in range(5):
            ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=states[s])

        ax.set_xlabel('P({0} | state) for agent 0 ({1})'.format(prob_state, pair[0]))
        ax.set_ylabel('P({0} | state) for agent 1 ({1})'.format(prob_state, pair[1]))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        set_legend_ax(ax, ncols=5)

        if c == cols - 2:
            break

    ax = axes[-1]
    pair_colours = [["red", "darkorange"], ["blue", "cyan"]]

    # Two agents
    for r, results in enumerate(ordered_results):
        av_R1, std_R1, _, _, av_R2, std_R2, _, _ = get_av_epoch_R_std_TFT(results)
        X = [[av_R1, std_R1], [av_R2, std_R2]]
        agent_pair = results["config"]["simulation"]["agent_pair"]

        pair = viewer_friendly_pair(agent_pair, as_list=True)

        for a in range(2):
            R = X[a]

            avr_v1 = moving_average(R[0], window_size=1)
            min_v1 = moving_average(R[0] - R[1], window_size=1)
            max_v1 = moving_average(R[0] + R[1], window_size=1)

            x = np.arange(np.shape(avr_v1)[0])
            ax.plot(x, avr_v1, pair_colours[r][a], alpha=0.5, label="a{0} {1}".format(a, pair[a]))
            ax.fill_between(x, min_v1, max_v1, color=pair_colours[r][a], alpha=0.1)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Average reward per step, R')
    set_legend_ax(ax)

    # plt.show()
    plt.savefig(filename_to_save)
    pass


def plot_policy_walk_through_space(results, intervals, states=["s0", "CC", "CD", "DC", "DD"],
                                   title="", filename_to_save="", prob_state="cooperation", top=[None, None]):
    cols = len(intervals)
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(14, 4), sharey=True, sharex=True)
    plt.subplots_adjust(left=0.07, right=0.97, top=0.79, bottom=0.14, wspace=0.1, hspace=0.27)

    colors = ["cyan", "blue", "orange", "green", "red"]

    X = get_epoch_policies(results)
    X = X[top[0]:top[1]]
    agent_pair = results["config"]["simulation"]["agent_pair"]
    pair = viewer_friendly_pair(agent_pair, as_list=True)

    for c, ax in enumerate(axes):
        start = intervals[c][0]
        depth = intervals[c][1]

        # PLot where the policies end up
        for s in range(5):
            ax.scatter(X[:, depth, 0, s], X[:, depth, 1, s], s=20, c=colors[s], alpha=0.5, label=states[s])

        # walk through space
        for repeat in X:
            for s in range(5):
                ax.plot(repeat[start:depth+1, 0, s], repeat[start:depth+1, 1, s], c=colors[s], alpha=0.25)

        ax.set_title("[{0}:{1}]".format(start, depth))
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        # set_legend_ax(ax, ncols=5)

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.95), ncol=5, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'P({0} | state) for agent 0 ({1})'.format(prob_state, pair[0]), ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'P({0} | state) for agent 1 ({1})'.format(prob_state, pair[1]), va='center', rotation='vertical', fontsize=12)

    # plt.show()
    plt.savefig(filename_to_save)
    pass



if __name__ == "__main__":
    # plot_single_sim_run("results/lolaom_ST_space/S01xT08/result_lolaom_vs_lolaom_IPD.json")
    plot_R_std_TFT_through_epochs("results/lola1_random_init_policy_robustness/R25/lola1_vs_lola1_IPD.json")
    # plot_average_value("results/result_lolaom_vs_lolaom_IPD.json")
