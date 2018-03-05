import numpy as np
import os
import glob
import fnmatch
import json
import matplotlib.pyplot as plt


def find_files(directory, pattern, ignore_root=True):
    for root, dirs, files in os.walk(directory):
        if not (root == directory and ignore_root):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename


def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_policies(results):
    policies = []
    for experiment in results["results"]["seeds"]:
        policies.append([experiment["P1"], experiment["P2"]])
    return policies


def get_collective_policies(folder, pattern='*.json'):
    results = dict()
    for filename in find_files(folder, pattern):
        X = get_policies(load_results(filename))
        results[filename] = X
    return results


def lolaom_dilemmas(folder="lolaom_dilemmas/"):
    game = "IPD"
    # game = "ISD"
    # game = "ISH"
    results = get_collective_policies(folder, "*{0}.json".format(game))

    nums = [25, 50, 75, 100]
    lengths = [20, 50, 100, 150]
    keys = [["n={0}, l={1}".format(n, l) for l in lengths] for n in nums]

    sorted_results = [[None for _ in lengths] for _ in nums]

    def index(filename):
        folder = filename.split('/')[1]
        num_len = folder.split('x')
        return nums.index(int(num_len[0])), lengths.index(int(num_len[1]))

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = X

    plot_policies(np.array(sorted_results), keys, "How the number and length of rollouts affects the final policy "
                                                  "of the agents in the {0} game".format(game))
    #
    # def swap(nparray):
    #     temp = np.copy(nparray[:, :, :, :, 1, 2])
    #     nparray[:, :, :, :, 1, 2] = nparray[:, :, :, :, 1, 3]
    #     nparray[:, :, :, :, 1, 3] = temp
    #     return nparray
    #
    # variance  = np.var([ipd_results, ish_results, isd_results], axis=3)
    # mean = np.mean([ipd_results, ish_results, isd_results], axis=3)
    #
    # [variance, mean] = swap(np.array([variance, mean]))
    #
    # print(ipd_results)


def plot_policies(results, keys, title, show=True, figsize=(13, 8), colours=None):
    rows = len(keys)
    cols = len(keys[0])

    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=figsize)
    fig.text(0.5, 0.96, title, ha='center', fontsize=14)
    fig.text(0.5, 0.02, 'P(cooperation | state) for agent 0', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 'P(cooperation | state) for agent 1', va='center', rotation='vertical', fontsize=12)

    colors = ["purple", "blue", "orange", "green", "red"]
    state = ["s0", "CC", "CD", "DC", "DD"]
    for r, row in enumerate(axes):
        for c, ax in enumerate(row):
            X = results[r][c]
            for s in range(5):
                ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=state[s])
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
        plt.savefig(title+".pdf")


def lolaom_ST_space(folder="lolaom_ST_space/"):
    game = "IPD"
    results = get_collective_policies(folder, "*{0}.json".format(game))

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
        folder = filename.split('/')[1]
        s_t = folder.split('x')
        return int(s_t[0][1:]), int(s_t[1][1:])

    for filename, X in results.items():
        i, j = index(filename)
        if game in filename:
            sorted_results[i][j] = np.array(X)

    plot_policies(np.array(sorted_results), keys, "How the S and T affect the final policy "
                                                  "of the agents in the {0} game".format(game),
                  show=False, figsize=(30, 30), colours=colours)


if __name__ == "__main__":
    lolaom_dilemmas()
    # lolaom_ST_space()
    pass