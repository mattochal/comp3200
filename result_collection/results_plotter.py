import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from result_collection.helper_func import *
from result_collection.benchmark_metrics import *


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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), shadow=True, ncol=ncols)


def plot_policies_and_v_timeline(ordered_results, states=["s0", "CC", "CD", "DC", "DD"], title="", filename_to_save="",
                                 prob_state="cooperation"):
    cols = len(ordered_results) + 1
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(11, 5))
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


def plot_policy_walk_through_space(results, intervals, states=["s0", "CC", "CD", "DC", "DD"], show=True, figsize=(12, 4),
                                   title="", filename_to_save="", prob_state="C", top=[None, None]):
    cols = len(intervals)
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=figsize, sharey=True, sharex=True)
    plt.subplots_adjust(left=0.09, right=0.97, top=0.79, bottom=0.16, wspace=0.1, hspace=0.27)

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
        m = 0.05
        ax.set_xlim([0-m, 1+m])
        ax.set_ylim([0-m, 1+m])
        # set_legend_ax(ax, ncols=5)

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.97), ncol=5, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'P({0}|s) for agent 0 ({1})'.format(prob_state, pair[0]), ha='center', fontsize=12)
    fig.text(0.015, 0.5, 'P({0}|s) for agent 1 ({1})'.format(prob_state, pair[1]), va='center', rotation='vertical', fontsize=12)

    # if show:
    #     plt.show()
    # else:
    plt.savefig(filename_to_save)
    pass


def plot_v_timelines_for_delta_eta_seperate_plots(ordered_results, states = ["s0", "CC", "CD", "DC", "DD"], title ="", titles = [], filename ="", prob_state ="cooperation", show=True, figsize=(10, 15)):
    nrows = len(ordered_results)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=figsize, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.08, wspace=0.22, hspace=0.27)

    # colors = ["cyan", "blue", "orange", "green", "red"]
    pair_colours = ["blue", "cyan"]

    agent_pair = ordered_results[0][0]["config"]["simulation"]["agent_pair"]
    pair = viewer_friendly_pair(agent_pair, as_list=True)

    # Two agents
    for r, ax in enumerate(axes):
        etas_x = []
        Y1 = []
        Y2 = []

        for c, results in enumerate(ordered_results[r]):
            eta = results["config"]["agent_pair"]["eta"]
            etas_x.append(eta)

            av_R1, std_R1, av_R2, std_R2, _ , _ = get_av_end_R_conf_TFT(results)
            Y1.append([av_R1, std_R1])
            Y2.append([av_R2, std_R2])

        Y1 = np.array(Y1)
        Y2 = np.array(Y2)

        ax.plot(etas_x, Y1[:, 0], pair_colours[0], alpha=0.5, label="a{0} {1}".format(0, pair[0]))
        ax.fill_between(etas_x, Y1[:, 0] + Y1[:, 1], Y1[:, 0] - Y1[:, 1], color=pair_colours[0], alpha=0.1)

        ax.plot(etas_x, Y2[:, 0], pair_colours[1], alpha=0.5, label="a{0} {1}".format(1, pair[1]))
        ax.fill_between(etas_x, Y2[:, 0] + Y2[:, 1], Y2[:, 0] - Y2[:, 1], color=pair_colours[1], alpha=0.1)

        ax.errorbar(etas_x, Y1[:, 0], yerr=Y1[:, 1], fmt='o', ecolor='g', capthick=2, label="a{0} {1}".format(0, pair[0]))
        ax.errorbar(etas_x, Y2[:, 0], yerr=Y2[:, 1], fmt='o', ecolor='g', capthick=2, label="a{0} {1}".format(1, pair[1]))

        # ax.scatter([eta], avr_v1, c=pair_colours[a], alpha=0.5, label="a{0} {1}".format(a, pair[a]))
        # ax.fill_between(x, min_v1, max_v1, color=pair_colours[a], alpha=0.1)
        # delta = ordered_results[0][0]["config"]["agent_pair"]["delta"]
        ax.set_title(titles[r])
        ax.set_ylabel('Average reward per step, R')

    handles, labels = ax.get_legend_handles_labels()
    # print(labels[:2])
    legend = fig.legend(handles[:2], labels[:2], loc='upper center', bbox_to_anchor=(0.51, 0.95), ncol=5, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.02, 0.5, r'$\delta', ha='center', fontsize=12)
    fig.text(0.5, 0.02, r'$\eta', va='center', fontsize=12)

    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass


def plot_v_timelines_for_delta_eta_combined_plot(ordered_results, states = ["s0", "CC", "CD", "DC", "DD"], title ="",
                                                 titles = [], filename ="", prob_state ="cooperation", show=True, figsize=(10, 10)):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.08, wspace=0.22, hspace=0.27)

    # colors = ["cyan", "blue", "orange", "green", "red"]
    pair_colours = ["purple", "blue", "orange", "green", "red", "violet", "lightblue", "gold", "lightgreen", "darkred"]

    agent_pair = ordered_results[0][0]["config"]["simulation"]["agent_pair"]
    pair = viewer_friendly_pair(agent_pair, as_list=True)

    # Two agents
    for r, row_results in enumerate(ordered_results):
        etas_x = []
        Y1 = []

        for c, results in enumerate(row_results):
            eta = results["config"]["agent_pair"]["detla"]
            etas_x.append(eta)

            av_R1, std_R1, _, _, _ , _ = get_av_end_R_conf_TFT(results, joint=True)
            Y1.append([av_R1, std_R1])

        Y1 = np.array(Y1)

        ax.plot(etas_x, Y1[:, 0], pair_colours[r], alpha=0.5, label="{0}".format(titles[r]))
        # ax.fill_between(etas_x, Y1[:, 0] + Y1[:, 1], Y1[:, 0] - Y1[:, 1], color=pair_colours[0], alpha=0.1)
        # ax.plot(etas_x, Y2[:, 0], pair_colours[1], alpha=0.5, label="a{0} {1}".format(1, pair[1]))
        # ax.fill_between(etas_x, Y2[:, 0] + Y2[:, 1], Y2[:, 0] - Y2[:, 1], color=pair_colours[1], alpha=0.1)

        ax.errorbar(etas_x, Y1[:, 0], yerr=Y1[:, 1], c=pair_colours[r],  ecolor=pair_colours[r], capthick=15)
        # ax.errorbar(etas_x, Y2[:, 0], yerr=Y2[:, 1], fmt='o', ecolor='g', capthick=2, label="a{0} {1}".format(1, pair[1]))

        # ax.scatter([eta], avr_v1, c=pair_colours[a], alpha=0.5, label="a{0} {1}".format(a, pair[a]))
        # ax.fill_between(x, min_v1, max_v1, color=pair_colours[a], alpha=0.1)
        # delta = ordered_results[0][0]["config"]["agent_pair"]["delta"]
        ax.set_title(titles[r])
        ax.set_ylabel('Average reward per step, R')

    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.95), ncol=5, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.02, 0.5, "Average reward per step, R", va='center', fontsize=12)
    fig.text(0.5, 0.02, r'$\delta$', ha='center', fontsize=12)

    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass


def plot_delta_eta_row_col_benchmarks(ordered_results, title="", titles=[], filename="", show=True, figsize=(6, 8),
                                      delta_or_eta="eta"):
    n_metrix = 4

    fig, axes = plt.subplots(nrows=n_metrix, ncols=1, figsize=figsize, sharex=True)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.08, wspace=0.22, hspace=0.27)

    # colors = ["cyan", "blue", "orange", "green", "red"]
    # pair_colours = ["purple", "blue", "orange", "green", "red", "violet", "lightblue", "gold", "lightgreen", "darkred"]

    # agent_pair = ordered_results[0][0]["config"]["simulation"]["agent_pair"]
    # pair = viewer_friendly_pair(agent_pair, as_list=True)

    ax = axes[0]
    x = []

    # Average reward
    for c, results in enumerate(ordered_results):
        d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(d_or_e)

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: R(x1, x2, gamma=gamma, r1=p1, r2=p2))

        ax.errorbar(d_or_e, R1, yerr=conf_R1, fmt='o', c="darkgreen", ecolor="green", capthick=15, alpha=0.75)

    # ax.set_xlabel(r'$\{0}$'.format(delta_or_eta))
    ax.set_title(r'Effect of $\{0}$ on the average reward per step, R'.format(delta_or_eta))
    ax.set_ylabel('Average reward per step, R')
    ax.set_yticks(np.linspace(-2, -1, 5))
    ax.grid(True)

    ax = axes[1]
    # Convergence
    for c, results in enumerate(ordered_results):
        d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(d_or_e)

        epoch_policies = np.array(get_epoch_policies(results))
        (conv1, conf_conv1) = get_av_metrics_for_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: conv_2p(x1, x2, x=0.90, game="IPD"))

        ax.errorbar(d_or_e, conv1, yerr=conf_conv1, fmt='s', c="darkgreen", ecolor="green", capthick=15, alpha=0.75, label="Conv(90%)")

        (conv1, conf_conv1) = get_av_metrics_for_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                               join_policies=True,
                                                               conf_interval=0.95,
                                                               metric_fn=lambda x1, x2: conv_2p(x1, x2, x=0.95,
                                                                                                game="IPD"))

        ax.errorbar(d_or_e, conv1, yerr=conf_conv1, fmt='o', c="darkblue", ecolor="blue", capthick=15, alpha=0.75, label="Conv(95%)")

        (conv1, conf_conv1) = get_av_metrics_for_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                               join_policies=True,
                                                               conf_interval=0.95,
                                                               metric_fn=lambda x1, x2: conv_2p(x1, x2, x=0.99,
                                                                                                game="IPD"))

        ax.errorbar(d_or_e, conv1, yerr=conf_conv1, fmt='x', c="darkred", ecolor="red", capthick=15, alpha=0.75, label="Conv(99%)")

        # set_legend_ax(ax, 3)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[:3], labels[:3], loc='lower left', ncol=1, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)
    # ax.set_xlabel(r'$\{0}$'.format(delta_or_eta))
    ax.set_title(r'Effect of $\{0}$ on convergence, Conv(x), to x% of TFT strategy.'.format(delta_or_eta))
    ax.set_ylabel('Average convergence time')
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 300, 4))

    ax = axes[2]
    x = []

    # Average reward
    for c, results in enumerate(ordered_results):
        d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(d_or_e)

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: tft2(x1, x2))

        ax.errorbar(d_or_e, R1, yerr=conf_R1, fmt='o', c="darkgreen", ecolor="green", capthick=15, alpha=0.75)

    # ax.set_xlabel(r'$\{0}$'.format(delta_or_eta))
    ax.set_title(r'Effect of $\{0}$ on %TFT2'.format(delta_or_eta))
    ax.set_ylabel('Average %TFT2')
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 1, 6))

    ax = axes[3]
    x = []

    # Average reward
    for c, results in enumerate(ordered_results):
        d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(d_or_e)

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=1))

        ax.errorbar(d_or_e, R1, yerr=conf_R1, fmt='o', c="darkblue", ecolor="blue", capthick=15, alpha=0.55, label="CC")

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=2))

        ax.errorbar(d_or_e, R1, yerr=conf_R1, fmt='s', c="orange", ecolor="gold", capthick=15, alpha=0.75, label="CD")

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=3))

        ax.errorbar(d_or_e, R1, yerr=conf_R1, fmt='x', c="darkgreen", ecolor="green", capthick=15, alpha=0.55, label="DC")

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=4))

        ax.errorbar(d_or_e, R1, yerr=conf_R1, fmt='^', c="darkred", ecolor="red", capthick=15, alpha=0.55, label="DD")

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[:4], labels[:4], loc='best', ncol=1, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    ax.set_xlabel(r'$\{0}$'.format(delta_or_eta))
    ax.set_title(r'Effect of $\{0}$ on expected number of visits to state s'.format(delta_or_eta))
    ax.set_ylabel('Expected visits to state s')
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 100, 6))

    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass


def plot_metrics_graph_randomness(ordered_results, title="", titles=[], filename="", show=True, figsize=(8, 3)):
    n_metrix = 4

    fig, axes = plt.subplots(nrows=n_metrix, ncols=1, figsize=figsize, sharex=True)
    plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.08, wspace=0.22, hspace=0.27)

    ax = axes[0]
    x = []

    # Average reward
    for c, results in enumerate(ordered_results):
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: R(x1, x2, gamma=gamma, r1=p1, r2=p2))

        ax.errorbar(x[c], R1, yerr=conf_R1, fmt='o', c="darkgreen", ecolor="green", capthick=15, alpha=0.75)

    # ax.set_xlabel(r'$\{0}$'.format(delta_or_eta))
    ax.set_xlabel('Randomness, r, in initial policy probability drawn from a uniform distribution of [0.5-r, 0.5+r]')
    ax.set_title(r'Effect of r on the average reward per step, R')
    ax.set_ylabel('Av. reward per step, R')
    ax.set_yticks(np.linspace(-2, -1, 5))
    ax.grid(True)

    ax = axes[1]
    # Convergence
    for c, results in enumerate(ordered_results):
        # d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(x[c])

        epoch_policies = np.array(get_epoch_policies(results))
        (conv1, conf_conv1) = get_av_metrics_for_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                               join_policies=True,
                                                               conf_interval=0.95,
                                                               metric_fn=lambda x1, x2: conv_2p(x1, x2, x=0.90,
                                                                                                game="IPD"))

        ax.errorbar(x[c], conv1, yerr=conf_conv1, fmt='s', c="darkgreen", ecolor="green", capthick=15, alpha=0.75,
                    label="Conv(90%)")

        (conv1, conf_conv1) = get_av_metrics_for_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                               join_policies=True,
                                                               conf_interval=0.95,
                                                               metric_fn=lambda x1, x2: conv_2p(x1, x2, x=0.95,
                                                                                                game="IPD"))

        ax.errorbar(x[c], conv1, yerr=conf_conv1, fmt='o', c="darkblue", ecolor="blue", capthick=15, alpha=0.75,
                    label="Conv(95%)")

        (conv1, conf_conv1) = get_av_metrics_for_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                               join_policies=True,
                                                               conf_interval=0.95,
                                                               metric_fn=lambda x1, x2: conv_2p(x1, x2, x=0.99,
                                                                                                game="IPD"))

        ax.errorbar(x[c], conv1, yerr=conf_conv1, fmt='x', c="darkred", ecolor="red", capthick=15, alpha=0.75,
                    label="Conv(99%)")

        # set_legend_ax(ax, 3)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[:3], labels[:3], loc='lower left', ncol=1, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    ax.set_xlabel('Randomness, r, in initial policy probability drawn from a uniform distribution of [0.5-r, 0.5+r]')
    ax.set_title(r'Effect of r on convergence, Conv(x), to x% of TFT strategy.')
    ax.set_ylabel('Average convergence time')
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 300, 4))

    ax = axes[2]
    x = []

    # Average reward
    for c, results in enumerate(ordered_results):
        # d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(x[c])

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: tft2(x1, x2))

        ax.errorbar(x[c], R1, yerr=conf_R1, fmt='o', c="darkgreen", ecolor="green", capthick=15, alpha=0.75)

    ax.set_xlabel('Randomness, r, in initial policy probability drawn from a uniform distribution of [0.5-r, 0.5+r]')
    ax.set_title(r'Effect of r on %TFT2')
    ax.set_ylabel('Average %TFT2')
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 1, 6))

    ax = axes[3]
    x = []

    # Average reward
    for c, results in enumerate(ordered_results):
        # d_or_e = results["config"]["agent_pair"][delta_or_eta]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        x.append(x[c])

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=1))

        ax.errorbar(x[c], R1, yerr=conf_R1, fmt='o', c="darkblue", ecolor="blue", capthick=15, alpha=0.55, label="CC")

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=2))

        ax.errorbar(x[c], R1, yerr=conf_R1, fmt='s', c="orange", ecolor="gold", capthick=15, alpha=0.75, label="CD")

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=3))

        ax.errorbar(x[c], R1, yerr=conf_R1, fmt='x', c="darkgreen", ecolor="green", capthick=15, alpha=0.55,
                    label="DC")

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: exp_s(x1, x2, state=4))

        ax.errorbar(x[c], R1, yerr=conf_R1, fmt='^', c="darkred", ecolor="red", capthick=15, alpha=0.55, label="DD")

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[:4], labels[:4], loc='best', ncol=1, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    ax.set_xlabel('Randomness, r, in initial policy probability drawn from a uniform distribution of [0.5-r, 0.5+r]')
    # ax.set_title(r'Effect of $\{0}$ on expected number of visits to state s'.format(delta_or_eta))
    ax.set_ylabel('Expected visits to state s')
    ax.grid(True)
    ax.set_yticks(np.linspace(0, 100, 6))

    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass

    ax = axes[0]
    x = np.linspace(0, 0.5, len(ordered_results))[:-1]
    # Average reward
    for c, x_pt in enumerate(x):
        results = ordered_results[c]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: R(x1, x2, gamma=gamma, r1=p1, r2=p2))

        ax.errorbar(x_pt, R1, yerr=conf_R1, fmt='o', c="darkgreen", ecolor="green", capthick=15, alpha=0.75)

    ax.set_xlabel('Randomness, r, in initial policy probability drawn from a uniform distribution of [0.5-r, 0.5+r]')
    ax.set_title(r'Effect of noise in the initial policy on av. reward per step')
    ax.set_ylabel('Av. reward per step, R')

    ax = axes[1]
    x = []

    # Average tft
    for c, x_pt in enumerate(x):
        results = ordered_results[c]
        gamma = results["config"]["agent_pair"]["gamma"]
        p1 = results["config"]["game"]["payoff1"]
        p2 = results["config"]["game"]["payoff2"]

        end_policies = np.array(get_end_policies(results))
        (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                         conf_interval=0.95,
                                                         metric_fn=lambda x1, x2: tft2(x1, x2))

        ax.errorbar(x_pt, R1, yerr=conf_R1, fmt='o', c="darkgreen", ecolor="green", capthick=15, alpha=0.75)

    # ax.set_xlabel(r'$\{0}$'.format(delta_or_eta))
    ax.set_ylabel('Average %TFT')

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles[:4], labels[:4], loc='best', ncol=1, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    # ax.grid(True)
    # ax.set_yticks(np.linspace(0, 100, 6))

    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass


def plot_metrics_timeline(single_results, metrics=[], ylabels=[], xlabels=[], ybounds=[], filename="", yticks=[], show=True, figsize=(6, 8), n_metrics = None):

    fig, axes = plt.subplots(nrows=n_metrics, ncols=1, figsize=figsize)

    # Two agents
    for r, metric in enumerate(metrics):
        epoch_policies = np.array(get_epoch_policies(single_results))
        av_conf_results = get_av_metrics_for_epoch_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                                  join_policies=metric[4],
                                                                  conf_interval=0.95,
                                                                  metric_fn=metric[1])

        if not metric[4]:
            ax = axes[metric[2]]
            for l, label in enumerate(metric[0]):
                value_fn = av_conf_results[:, 0+l]
                conf = av_conf_results[:, 2+l]
                x = np.arange(np.shape(value_fn)[0])
                ax.plot(x, value_fn, c=metric[3][l], alpha=0.5, label=label)
                ax.fill_between(x, value_fn- conf, value_fn + conf, color=metric[3][l], alpha=0.2)
        else:
            value_fn = av_conf_results[:, 0]
            conf = av_conf_results[:, 1]
            x = np.arange(np.shape(value_fn)[0])

            ax = axes[metric[2]]
            ax.plot(x, value_fn, c=metric[3][0], alpha=0.5, label=metric[0][0])
            ax.fill_between(x, value_fn - conf, value_fn + conf, color=metric[3][0], alpha=0.2)

    for a, ax in enumerate(axes):
        ax.set_ylabel(ylabels[a])
        set_legend_ax(ax, 4)
        ax.set_xlabel('Iterations')
        ax.set_yticks(yticks[a])
        ax.set_ylim(ybounds[a])

    plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.08, wspace=0.18, hspace=0.38)
    # show = True
    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass


def plot_metrics_across_x(single_results, metrics=[], ylabels=[], xlabels=[], xticks= [], yticks=[], ybounds = [], filename="", show=True, figsize=(8, 10), n_metrics = None):

    fig, axes = plt.subplots(nrows=n_metrics, ncols=1, figsize=figsize)

    # Two agents
    for r, metric in enumerate(metrics):
        epoch_policies = single_results
        av_conf_results = get_av_metrics_for_epoch_policy_arrays(epoch_policies[:, :, 0], epoch_policies[:, :, 1],
                                                                  join_policies=metric[4],
                                                                  conf_interval=0.95,
                                                                  metric_fn=metric[1])

        if not metric[4]:
            ax = axes[metric[2]]
            for l, label in enumerate(metric[0]):
                value_fn = av_conf_results[:, 0+l]
                conf = av_conf_results[:, 2+l]
                # x = np.arange(np.shape(value_fn)[0])
                ax.plot(xticks, value_fn, c=metric[3][l], alpha=0.5, label=label)
                ax.fill_between(xticks, value_fn- conf, value_fn + conf, color=metric[3][l], alpha=0.2)
        else:
            value_fn = av_conf_results[:, 0]
            conf = av_conf_results[:, 1]
            # x = np.arange(np.shape(value_fn)[0])

            ax = axes[metric[2]]
            ax.plot(xticks, value_fn, c=metric[3][0], alpha=0.5, label=metric[0][0])
            ax.fill_between(xticks, value_fn - conf, value_fn + conf, color=metric[3][0], alpha=0.2)

    reduced_xticks = np.linspace(0,0.5, int(len(xticks)/5)+1)
    for a, ax in enumerate(axes):
        ax.set_ylabel(ylabels[a])
        set_legend_ax(ax, 4)
        ax.set_xlabel(xlabels[a])
        ax.set_xticks(xticks, minor=True)
        ax.set_xticks(reduced_xticks)
        ax.set_yticks(yticks[a])
        ax.set_ylim(ybounds[a])

    plt.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.08, wspace=0.18, hspace=0.38)
    # show = True
    if show:
        plt.show()
    else:
        plt.savefig(filename)
    pass

if __name__ == "__main__":
    # plot_single_sim_run("results/lolaom_ST_space/S01xT08/result_lolaom_vs_lolaom_IPD.json")
    # plot_R_std_TFT_through_epochs("results/lola1_random_init_policy_robustness/R25/lola1_vs_lola1_IPD.json")
    # plot_average_value("results/result_lolaom_vs_lolaom_IPD.json")
    plot_R_std_TFT_through_epochs("../results/lola_robust_delta_eta/D00/E00/lola1_vs_lola1_IPD.json")
