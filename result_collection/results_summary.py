from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import fnmatch
import json
from result_collection.helper_func import *
from result_collection.helper_pca import *


def text_summary_TFT_R_std_from_folder(path):
    for filename in find_files(path, "*.json"):
        print(filename)
        text_summary_TFT_R_std_single(filename)


def text_summary_TFT_R_std_single(path):
    results = load_results(path)
    av_reward_1, std_av_reward_1, av_reward_2, std_av_reward_2, av_TFT_percent_1, av_TFT_percent_2 = \
        get_av_end_R_std_TFT(results)

    print("R(std): {:.2f}({:.2f}), {:.2f}({:.2f}), TFT%: {:.1f}, {:.1f}".format(av_reward_1, std_av_reward_1,
                                                                                av_reward_2, std_av_reward_2,
                                                                                av_TFT_percent_1 * 100,
                                                                                av_TFT_percent_2 * 100))


def plot_TFT(path):
    results = load_results(path)
    X = np.array(get_initial_policies(results))
    _, _, TFT1, TFT2 = get_end_R_std_compare(results)

    labels = []
    for t in TFT1:
        if t >= 0.9:
            labels.append("b")
        else:
            labels.append("r")

    for i, t in enumerate(TFT2):
        if t < 0.9:
            labels[i] = "r"

    Y = np.concatenate((X[:, 0, :], X[:, 1, :]), 1)
    visualise_2D_with_lda(Y, labels)



def plot_decistion_tree_TFT(path):
    results = load_results(path)
    X = np.array(get_initial_policies(results))
    _, _, TFT1, TFT2 = get_end_R_std_compare(results)

    labels = []
    for t in TFT1:
        if t >= 0.9:
            labels.append("b")
        else:
            labels.append("r")

    for i, t in enumerate(TFT2):
        if t < 0.9:
            labels[i] = "r"

    Y = np.concatenate((X[:, 0, :], X[:, 1, :]), 1)
    visualise_decision(Y, labels)


if __name__ == "__main__":
    # text_summary_TFT_R_std_from_folder("results/lola_random_init_long_epochs/")
    plot_decistion_tree_TFT("results/lola_random_init_long_epochs/E06xD06/lola_vs_lola_IPD.json")
    # plot_average_value("results/result_lolaom_vs_lolaom_IPD.json")

