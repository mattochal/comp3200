import numpy as np
import matplotlib.pyplot as plt
import re

from result_collection.helper_func import *


def plot_2ax_policies(results, keys, title, show=True, figsize=(13, 8), colours=None, filename=None):
    rows = np.shape(results)[0]
    cols = np.shape(results)[1]

    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=figsize)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'P(cooperation | state) for agent 0', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'P(cooperation | state) for agent 1', va='center', rotation='vertical', fontsize=12)

    colors = ["cyan", "blue", "orange", "green", "red"]
    state = ["s0", "CC", "CD", "DC", "DD"]
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            X = results[r][c]
            for s in range(5):
                ax.scatter(X[:, 0, s], X[:, 1, s], s=25, c=colors[s], alpha=0.5, label=state[s])
            ax.set_title(keys[r][c], fontsize=11)
            if colours is not None:
                ax.set_facecolor(colours[r][c])

    plt.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.08, wspace=0.07, hspace=0.27)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.95), ncol=5, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    if show:
        plt.show()
    else:
        if filename is None:
            filename = title
        plt.savefig(filename+".pdf")


def plot_1ax_policies(results, keys, title, show=True, figsize=(13, 8), colours=None, filename=None):
    rows = np.shape(results)[0]

    fig, axes = plt.subplots(ncols=rows, nrows=1, sharex=True, sharey=True, figsize=figsize)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'P(cooperation | state) for agent 0', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'P(cooperation | state) for agent 1', va='center', rotation='vertical', fontsize=12)

    colors = ["purple", "blue", "orange", "green", "red"]
    state = ["s0", "CC", "CD", "DC", "DD"]
    for r, ax in enumerate(axes):
        X = results[r]
        for s in range(5):
            ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=state[s])
        ax.set_title(keys[r], fontsize=11)
        if colours is not None:
            ax.set_facecolor(colours[r])

    plt.subplots_adjust(left=0.07, right=0.99, top=0.87, bottom=0.08, wspace=0.07, hspace=0.27)
    handles, labels = ax.get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.95), ncol=5, borderaxespad=0, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    if show:
        plt.show()
    else:
        if filename is None:
            filename = title
        plt.savefig(filename+".pdf")


def plot_1ax_R_std_TFT_through_epochs(results, keys, title, show=True, figsize=(13, 8), filename=None):
    rows = np.shape(results)[-1]

    fig, axes = plt.subplots(nrows=rows, ncols=1, sharex=True, sharey=True, figsize=figsize)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'Randomness, r, in initial policy parameter values \n'
                        'drawn from a uniform distribution of [0.5-r/100, 0.5+r/100]', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'Epochs', va='center', rotation='vertical', fontsize=12)

    # X = [ [[av_R1, std_R1, av_TFT1], [av_R2, std_R2, av_TFT2]],
    #       [[av_R1, std_R1, av_TFT1], [av_R2, std_R2, av_TFT2]], ... ]

    colors = ["b", "r"]
    labels = ["R", "TFT"]
    symbols = ["x", "+"]

    for r, ax in enumerate(axes):
        X = results[:, :, :, r]

        rt = 0
        # Two agents
        for a in range(2):
            R = X[:, a]

            avr_v1 = moving_average(R[:, 0], window_size=1)
            min_v1 = moving_average(R[:, 0] - R[:, 1], window_size=1)
            max_v1 = moving_average(R[:, 0] + R[:, 1], window_size=1)

            x = np.arange(np.shape(avr_v1)[0])
            ax.plot(x, avr_v1, colors[rt] + symbols[a], alpha=0.5, label="Agent " + str(a))
            ax.fill_between(x, min_v1, max_v1, color=colors[rt], alpha=0.1)

        ax.set_ylabel(labels[rt])
        ax.set_title(keys[r])
        ax.tick_params('y', colors=colors[rt])

        ax2 = ax.twinx()
        rt = 1
        for a in range(2):
            R = X[:, a]
            avr_v1 = moving_average(R[:, 2], window_size=1)

            x = np.arange(np.shape(avr_v1)[0])
            ax2.plot(x, avr_v1, colors[rt] + symbols[a], alpha=0.5)

        ax2.set_ylabel(labels[rt])
        ax2.tick_params('y', colors=colors[rt])

    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.04, wspace=0.07, hspace=0.27)
    handles, labels = ax.get_legend_handles_labels()

    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.51, 0.95),
                        ncol=5, borderaxespad=0, fancybox=True)

    frame = legend.get_frame()
    frame.set_edgecolor('black')
    frame.set_alpha(1)

    if show:
        plt.show()
    else:
        if filename is None:
            filename = title
        plt.savefig(filename + ".pdf")


def lolaom_dilemmas(folder="results/lolaom_dilemmas/"):
    game = "IPD"
    # game = "ISD"
    # game = "ISH"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    nums = [25, 50, 75, 100]
    lengths = [20, 50, 100, 150]
    keys = [["n={0}, l={1}".format(n, l) for l in lengths] for n in nums]

    sorted_results = [[None for _ in lengths] for _ in nums]

    def index(filename):
        folder = filename.split('/')[2]
        num_len = folder.split('x')
        return nums.index(int(num_len[0])), lengths.index(int(num_len[1]))

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = X

    plot_2ax_policies(np.array(sorted_results), keys, "How the number and length of rollouts affects the final policy "
                                                  "of the agents in the {0} game".format(game))


def lolaom_ST_space(folder="results/lolaom_ST_space/"):
    game = "IPD"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    keys = [["S={0:.2f}, T={1:.2f}".format(s, t) for t in T] for s in S]
    sorted_results = [[None for _ in T] for _ in S]

    color_dict = {"IPD": "#ffeaea", "ISH": "#edffea", "ISD": "#eafdff", "None": "#efefef"}
    colours = [[None for _ in T] for _ in S]
    for j, t in enumerate(T):
        for i, s in enumerate(S):
            if t > 1 and 0 > s and t + s < 2:
                d_type = "IPD"
            elif t > 1 > s >= 0 and t + s < 2:
                d_type = "ISD"
            elif 1 > t > 0 > s and t + s < 2:
                d_type = "ISH"
            else:
                d_type = "None"
            colours[i][j] = color_dict[d_type]

    def index(filename):
        folder = filename.split('/')[2]
        s_t = folder.split('x')
        return int(s_t[0][1:]), int(s_t[1][1:])

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = np.array(X)

    plot_2ax_policies(np.array(sorted_results), keys, "How the S and T affect the final policy "
                                                  "of the agents in across the dilemmas",
                      show=False, figsize=(30, 30), colours=colours)


def lolaom_rollouts_small(folder="results/lolaom_rollouts_small/"):
    game = "IPD"
    # game = "ISD"
    # game = "ISH"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    nums = [5, 10, 15, 20]
    lengths = [5, 10, 15, 20]
    keys = [["n={0}, l={1}".format(n, l) for l in lengths] for n in nums]

    sorted_results = [[None for _ in lengths] for _ in nums]

    def index(filename):
        folder = filename.split('/')[2]
        num_len = folder.split('x')
        return nums.index(int(num_len[0])), lengths.index(int(num_len[1]))

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = X

    plot_2ax_policies(np.array(sorted_results), keys, "How the number and length of rollouts affects the final policy "
                                                  "of the agents in the {0} game".format(game))


def lolaom_IPD_SG_space(folder="results/lolaom_IPD_SG_space/"):
    game = "IPD"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    S = np.linspace(-1.0, 0.0, num=9)
    Gammas = np.linspace(0.0, 1.0, num=11)
    Gammas[10] = 0.99

    keys = [["S={0:.2f}, gamma={1:.2f}".format(s, g) for g in Gammas] for s in S]
    sorted_results = [[None for _ in Gammas] for _ in S]

    def index(filename):
        folder = filename.split('/')[2]
        s_t = folder.split('x')
        return int(s_t[0][1:]), int(s_t[1][1:])

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = np.array(X)

    plot_2ax_policies(np.array(sorted_results), keys,
                  "How the S and gamma affects the final policy of the agents in the {0} game".format(game),
                      show=False, figsize=(30, 30))


def lolaom_policy_init(folder="results/lolaom_policy_init/"):
    game = "IPD"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    theta1 = np.linspace(0.0, 1.0, num=9)
    theta2 = np.linspace(0.0, 1.0, num=9)

    keys = [["theta2_i={0:.2f}, theta1_i={1:.2f}".format(t2, t1) for t1 in theta1] for t2 in theta2]
    sorted_results = [[None for _ in theta1] for _ in theta2]

    def index(filename):
        folder = filename.split('/')[2]
        s_t = folder.split('x')
        return int(s_t[0][1:]), int(s_t[1][1:])

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = np.array(X)

    plot_2ax_policies(np.array(sorted_results), keys,
                  "How the initial policy parameters affects the final policy of the agents in the {0} game".format(game),
                      show=False, figsize=(30, 30))


def lolaom_long_epochs(folder="results/lolaom_long_epochs/"):
    game = "IPD"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    ETA = [0.01, 0.1, 0.5, 1.0, 5, 7.5, 10, 15]
    DELTA = [0.0005, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 3.0]

    keys = [["E={0:.4f}, D={1:.4f}".format(e, d) for d in DELTA] for e in ETA]
    sorted_results = [[None for _ in DELTA] for _ in ETA]

    def index(filename):
        folder = filename.split('/')[2]
        e_d = folder.split('x')
        return int(e_d[0][1:]), int(e_d[1][1:])

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = np.array(X)

    plot_2ax_policies(np.array(sorted_results), keys,
                  "results/How delta and eta affect the final policy of the agents in the {0} game".format(game),
                      show=False, figsize=(30, 30))


def lolaom_random_init_long_epochs(folder="../results/lolaom_random_init_long_epochs/", agents="LOLAOM"):
    game = "IPD"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    ETA = [0.01, 0.1, 0.5, 1.0, 5, 7.5, 10, 15]
    DELTA = [0.0005, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 3.0]

    keys = [["E={0:.4f}, D={1:.4f}".format(e, d) for d in DELTA] for e in ETA]
    sorted_results = [[None for _ in DELTA] for _ in ETA]

    def index(filename):
        e_d = re.findall('[E|D]\d+', filename)
        return int(e_d[0][1:]), int(e_d[1][1:])

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = np.array(X)

    plot_2ax_policies(np.array(sorted_results), keys,
                  "../results/How delta and eta affect the final policy of the {1} agents in the {0} game "
                  "with random parameter initialisation".format(game, agents),
                      show=False, figsize=(40, 40), filename=folder)


def lola_random_init_long_epochs(folder="../results/lola_random_init_long_epochs/"):
    lolaom_random_init_long_epochs(folder, agents="LOLA")


def lola1_random_init_policy_robustness_500(folder="../results/lola1_random_init_policy_robustness/", agents="LOLA1"):
    game = "IPD"
    results = collect_experiment_ith_policies(folder, 500, "*{0}.json".format(game))

    randomness = np.linspace(0, 0.5, 51)

    keys = ["R=[{0:.2f}, {1:.2f}]".format(0.5-r, 0.5+r) for r in randomness]
    sorted_results = [None for _ in randomness]

    def index(filename):
        return int(re.findall('R\d+', filename)[0][1:])

    for filename, X in results.items():
        i = index(filename)
        if game in filename and i < len(sorted_results):
            sorted_results[i] = np.array(X)

    plot_1ax_policies(np.array(sorted_results), keys,
                  "How randomness affects the policy of the {1} agents in the {0} game after playing 500 iterations"
                  "with random parameter initialisation".format(game, agents),
                      show=False, figsize=(200, 5), filename=folder[:-1]+"_500")


def lola1_random_init_policy_robustness(folder="../results/lola1_random_init_policy_robustness/", agents="LOLA1"):
    game = "IPD"
    results = collect_experiment_end_policies(folder, "*{0}.json".format(game))

    randomness = np.linspace(0, 0.5, 51)

    keys = ["R=[{0:.2f}, {1:.2f}]".format(0.5-r, 0.5+r) for r in randomness]
    sorted_results = [None for _ in randomness]

    def index(filename):
        return int(re.findall('R\d+', filename)[0][1:])

    for filename, X in results.items():
        i = index(filename)
        if game in filename and i < len(sorted_results):
            sorted_results[i] = np.array(X)

    plot_1ax_policies(np.array(sorted_results), keys,
          "How randomness affects the policy of the {1} agents in the {0} game after playing 1000 iterations"
          "with random parameter initialisation".format(game, agents),
                      show=False, figsize=(200, 5), filename=folder[:-1])


def lola1_random_init_policy_robustness_through_epochs(folder="../results/lola1_random_init_policy_robustness/", agents="LOLA1"):
    game = "IPD"
    results = collect_experiment_epoch_R_std_TFT(folder, "*{0}.json".format(game))

    randomness = np.linspace(0, 0.5, 51)

    keys = ["After training for {0} epochs".format(r*50) for r in np.linspace(0, 20, 21)]
    sorted_results = [None for _ in randomness]

    def index(filename):
        return int(re.findall('R\d+', filename)[0][1:])

    for filename, X in results.items():
        i = index(filename)
        if game in filename and i < len(sorted_results):
            sorted_results[i] = np.array(X)

    plot_1ax_R_std_TFT_through_epochs(np.array(sorted_results), keys,"How randomness in policy parameter initialisation"
                                                                     " affects the end policy of the {1} agents in the "
                                                                     "{0} game".format(game, agents),
                                      show=False, figsize=(15, 40), filename=folder[:-1]+"_through_epochs")


def lola1b_random_init_policy_robustness(folder="../results/lola1b_random_init_policy_robustness/", agents="LOLA1B"):
    lola1_random_init_policy_robustness(folder, agents)


def lola1b_random_init_policy_robustness_500(folder="../results/lola1b_random_init_policy_robustness/", agents="LOLA1B"):
    lola1_random_init_policy_robustness_500(folder, agents)


def lola1b_random_init_policy_robustness_through_epochs(folder="../results/lola1b_random_init_policy_robustness/", agents="LOLA1B"):
    lola1_random_init_policy_robustness_through_epochs(folder, agents)


if __name__ == "__main__":
    # lolaom_dilemmas()
    # lolaom_ST_space()
    # lolaom_IPD_SG_space()
    # lolaom_policy_init()
    # lolaom_rollouts_small()
    # lolaom_random_init_long_epochs()
    # lola1_random_init_policy_robustness_500()
    # lola1b_random_init_policy_robustness()
    # lola1b_random_init_policy_robustness_500()
    lola1_random_init_policy_robustness_through_epochs()
    # lola1b_random_init_policy_robustness_through_epochs()
    pass