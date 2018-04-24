import numpy as np
import os
import fnmatch
import json
import numpy as np
import scipy as sp
import scipy.stats
from collections import defaultdict


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a, axis=0)
    h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


# Given directory to look in
# and a pattern in regex for file ending
# find the files matching files
# You can choose to ignore matches at the root of the specified directory
def find_files(directory, pattern, ignore_root=True):
    for root, dirs, files in os.walk(directory):
        if not (root == directory and ignore_root):
            for basename in files:
                if fnmatch.fnmatch(root + basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename


# Loads json results given path to json file
def load_results(path):
    print("Loading...", path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# Saves result to given filename
def save_results(results, filename):
    print("Saving...", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        json.dump(results, outfile, indent=2)


# Given single experiment results in json format
# extracts end policies for both agents
def get_end_policies(results):
    policies = []
    for experiment in results["results"]["seeds"]:
        policies.append([experiment["P1"], experiment["P2"]])
    return policies


# Given single experiment results in json format
# extracts initial policies for both agents
def get_initial_policies(results):
    policies = []
    for experiment in results["results"]["seeds"]:
        policies.append([experiment["epoch"][0]["P1"], experiment["epoch"][0]["P2"]])
    return policies


# Given single experiment results in json format
# extracts policies after ith iteration for both agents
def get_ith_policies(results, i):
    policies = []
    for experiment in results["results"]["seeds"]:
        policies.append([experiment["epoch"][i]["P1"], experiment["epoch"][i]["P2"]])
    return policies


# Given single experiment results in json format
# extracts the policies through epochs for both agents
def get_epoch_policies(results):
    ps = []
    for experiment in results["results"]["seeds"]:
        p = []
        for epoch in experiment["epoch"]:
            p.append([epoch["P1"], epoch["P2"]])
        ps.append(p)
    return np.array(ps)


# Given single experiment results in json format
# extracts the value function through epochs for both agents.
# If value function is not available, then it is calculated based on parameters
def get_epoch_value_fns(results):
    vs = []
    for experiment in results["results"]["seeds"]:
        v = []
        for epoch in experiment["epoch"]:
            if "V1" in epoch:
                v.append([epoch["V1"], epoch["V2"]])
            else:
                v.append([epoch["V1"], epoch["V2"]])
        vs.append(v)
    return np.array(vs)


# Given a directory to a folder containing multiple result files stored in json format
# return a dictionary keyed with file paths and the loaded policy
def collect_experiment_results(folder, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]: # + sorted(filenames)[10:top+10]:
        results[filename] = load_results(filename)
    return results


# Given a directory to a folder containing multiple result files stored in json format
# return a dictionary keyed with file paths and the loaded policy
def collect_experiment_configs(folder, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]: # + sorted(filenames)[10:top+10]:
        results[filename] = load_results(filename)["config"]
    return results

# Given a directory to a folder containing multiple result files stored in json format
# return a dictionary keyed with file paths and the loaded policy
def collect_experiment_end_policies(folder, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]:
        X = get_end_policies(load_results(filename))
        results[filename] = X
    return results


# Given a directory to a folder containing multiple result files stored in json format
# return a dictionary keyed with file paths and the loaded policy
def collect_experiment_epoch_policies(folder, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]:
        X = get_epoch_policies(load_results(filename))
        results[filename] = X
    return results


# Given a directory to a folder containing multiple result files stored in json format
# return a dictionary keyed with file paths and the loaded policy
def collect_experiment_ith_policies(folder, i, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]:
        X = get_ith_policies(load_results(filename), i)
        results[filename] = X
    return results


def collect_experiment_epoch_R_std_TFT(folder, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]:
        json_results = load_results(filename)

        av_R1, std_R1, av_TFT1, _, av_R2, std_R2, av_TFT2, _ = \
            get_av_epoch_R_std_TFT(json_results)

        results[filename] = np.array([[av_R1, std_R1, av_TFT1], [av_R2, std_R2, av_TFT2]])

    return results


def collect_experiment_end_R_std_TFT(folder, pattern='*.json', top=None, compare_policy=None, tolerance=0.5):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]:
        json_results = load_results(filename)

        av_R1, std_R1, av_R2, std_R2, av_compare_1, av_compare_2 = \
            get_av_end_R_std_TFT(json_results, compare_policy, tolerance)

        results[filename] = np.array([[av_R1, std_R1, av_compare_1], [av_R2, std_R2, av_compare_2]])

    return results

# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# over the repeat through all the iterations in each epoch
# and calculate and return an overall average for those values
def get_av_epoch_R_std_TFT(results):
    all_R_1, all_R_2, all_TFT_1, all_TFT_2 = get_epoch_R_std_TFT(results)
    std_R1 = np.std(all_R_1, 0)
    std_R2 = np.std(all_R_2, 0)
    av_R1 = np.mean(all_R_1, 0)
    av_R2 = np.mean(all_R_2, 0)
    std_TFT1 = np.std(all_TFT_1, 0)
    std_TFT2 = np.std(all_TFT_2, 0)
    av_TFT1 = np.mean(all_TFT_1, 0)
    av_TFT2 = np.mean(all_TFT_2, 0)
    return av_R1, std_R1, av_TFT1, std_TFT1, av_R2, std_R2, av_TFT2, std_TFT2


def get_av_epoch_R_confintrv_TFT(results):
    all_R_1, all_R_2, all_TFT_1, all_TFT_2 = get_epoch_R_std_TFT(results)
    std_R1 = mean_confidence_interval(all_R_2)
    std_R2 = mean_confidence_interval(all_R_2)
    av_R1 = np.mean(all_R_1, 0)
    av_R2 = np.mean(all_R_2, 0)
    std_TFT1 = np.std(all_TFT_1, 0)
    std_TFT2 = np.std(all_TFT_2, 0)
    av_TFT1 = np.mean(all_TFT_1, 0)
    av_TFT2 = np.mean(all_TFT_2, 0)
    return av_R1, std_R1, av_TFT1, std_TFT1, av_R2, std_R2, av_TFT2, std_TFT2


# Given a 1D numpy array and a window size
# Calculate a sliding window average / moving average
def moving_average(a, window_size=5):
    ret = np.cumsum(a, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


# Given a policy for both agents ([P1, P2]),
# immediate reward function, r1 and r2, and the discount factor gamma
# calculate and return the value function
def calculate_value_fn_from_policy(end_policy, r1, r2, gamma):
    # Average reward per time step
    x1 = end_policy[0]
    x2 = end_policy[1]
    P = np.stack((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2))).T

    I = np.eye(4)
    Zinv1 = np.linalg.inv(I - gamma * P[1:, :])
    Zinv2 = np.linalg.inv(I - gamma * P[1:, :])

    V1 = np.matmul(np.matmul(P[0, :], Zinv1), r1)
    V2 = np.matmul(np.matmul(P[0, :], Zinv2), r2)
    return V1, V2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# and the similarity to a given comparison policy (assumed TFT)
# as an average over all repeats
def get_av_end_R_std_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], tolerance=0.5, joint=False):
    av_R_1, av_R_2, av_compare_1, av_compare_2 = get_end_R_std_compare(results, comparison_policy, tolerance)

    if not joint:
        std_av_reward_1 = np.std(av_R_1)
        av_R_1 = np.mean(av_R_1)

        std_av_reward_2 = np.std(av_R_2)
        av_R_2 = np.mean(av_R_2)

        av_compare_1 = np.mean(av_compare_1)
        av_compare_2 = np.mean(av_compare_2)
    else:
        join_R = av_R_1 + av_R_2
        std_av_reward_1 = std_av_reward_2 = np.std(join_R)
        av_R_1 = av_R_2 = np.mean(join_R)

        av_compare_1 = np.mean(av_compare_1)
        av_compare_2 = np.mean(av_compare_2)

    return av_R_1, std_av_reward_1, av_R_2, std_av_reward_2, av_compare_1, av_compare_2


def get_av_end_R_conf_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], tolerance=0.5, joint=False):
    av_R_1, av_R_2, av_compare_1, av_compare_2 = get_end_R_std_compare(results, comparison_policy, tolerance)

    if not joint:
        std_av_reward_1 = mean_confidence_interval(av_R_1)
        std_av_reward_2 = mean_confidence_interval(av_R_2)

        av_R_1 = np.mean(av_R_1)
        av_R_2 = np.mean(av_R_2)

        av_compare_1 = np.mean(av_compare_1)
        av_compare_2 = np.mean(av_compare_2)
    else:
        std_av_reward_1 = std_av_reward_2 = mean_confidence_interval(av_R_1 + av_R_2)
        av_R_1 = av_R_2 = np.mean(av_R_1+av_R_2)
        av_compare_1 = av_compare_2 = np.mean(av_compare_1 + av_compare_2)

    return av_R_1, std_av_reward_1, av_R_2, std_av_reward_2, av_compare_1, av_compare_2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# and the similarity to a given comparison policy (assumed TFT)
# as an average over all repeats
def get_av_ith_R_std_TFT(results, i, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], tolerance=0.5):
    av_R_1, av_R_2, av_compare_1, av_compare_2 = get_ith_R_std_compare(results, i, comparison_policy, tolerance)

    std_av_reward_1 = np.std(av_R_1)
    av_R_1 = np.mean(av_R_1)

    std_av_reward_2 = np.std(av_R_2)
    av_R_2 = np.mean(av_R_2)

    av_compare_1 = np.mean(av_compare_1)
    av_compare_2 = np.mean(av_compare_2)

    return av_R_1, std_av_reward_1, av_R_2, std_av_reward_2, av_compare_1, av_compare_2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# and the similarity to a given comparison policy (assumed TFT)
# for each individual repeat
def get_end_R_std_compare(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], tolerance=0.5):

    # Get some info about the simulation experiment
    game = results["config"]["simulation"]["game"]
    agent_pair = results["config"]["simulation"]["agent_pair"]
    r1 = np.array(results["config"]["game"]["payoff1"])
    r2 = np.array(results["config"]["game"]["payoff2"])
    gamma = results["config"]["agent_pair"]["gamma"]

    # Metrics to record
    all_R_1 = []
    all_R_2 = []
    all_compare_1 = []
    all_compare_2 = []

    for experiment in results["results"]["seeds"]:
        end_policy1 = np.array(experiment["P1"])
        end_policy2 = np.array(experiment["P2"])

        # Calculated as an absolute difference between end policy and
        # comparison = 1 - np.mean(np.abs(end_policy - comparison_policy), 1)

        V1, V2, comparison1, comparison2 = get_R_std_compare_for_single_policy_pair(end_policy1, end_policy2, r1, r2,
                                                                                    gamma, comparison_policy, tolerance)

        # TFT likeliness percentage across both agents
        all_compare_1.append(comparison1)
        all_compare_2.append(comparison2)
        all_R_1.append(V1)
        all_R_2.append(V2)

    return all_R_1, all_R_2, all_compare_1, all_compare_2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# and the similarity to a given comparison policy (assumed TFT)
# for each individual repeat
def get_ith_R_std_compare(results, i, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], tolerance=0.5):

    # Get some info about the simulation experiment
    game = results["config"]["simulation"]["game"]
    agent_pair = results["config"]["simulation"]["agent_pair"]
    r1 = np.array(results["config"]["game"]["payoff1"])
    r2 = np.array(results["config"]["game"]["payoff2"])
    gamma = results["config"]["agent_pair"]["gamma"]

    # Metrics to record
    all_R_1 = []
    all_R_2 = []
    all_compare_1 = []
    all_compare_2 = []

    for experiment in results["results"]["seeds"]:
        end_policy1 = np.array(experiment["epoch"][i]["P1"])
        end_policy2 = np.array(experiment["epoch"][i]["P2"])

        V1, V2, comparison1, comparison2 = get_R_std_compare_for_single_policy_pair(end_policy1, end_policy2, r1, r2,
                                                                                    gamma, comparison_policy, tolerance)

        # TFT likeliness percentage across both agents
        all_compare_1.append(comparison1)
        all_compare_2.append(comparison2)
        all_R_1.append(V1)
        all_R_2.append(V2)

    return all_R_1, all_R_2, all_compare_1, all_compare_2


def get_R_std_compare_for_single_policy_pair(p1, p2, r1, r2, gamma, comparison_policy, tolerance):

    comparison_p = np.array(comparison_policy).transpose()
    ps = np.array([p1, p2]).transpose()

    comparison1 = np.mean( np.max(np.abs(ps - comparison_p), 1) < tolerance)
    comparison2 = np.mean( np.max(np.abs(ps - comparison_p), 1) < tolerance)

    end_policy = [p1, p2]
    V1, V2 = calculate_value_fn_from_policy(end_policy, r1, r2, gamma)

    return V1 * (1 - gamma), V2 * (1 - gamma), comparison1, comparison2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# over the repeat through all the iterations in each epoch
# and return the similarity to a given comparison policy (assumed TFT)
def get_epoch_R_std_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]], tolerance=0.5):

    # Get some info about the simulation experiment
    game = results["config"]["simulation"]["game"]
    agent_pair = results["config"]["simulation"]["agent_pair"]
    r1 = np.array(results["config"]["games"][game]["payoff1"])
    r2 = np.array(results["config"]["games"][game]["payoff2"])
    gamma = results["config"]["agent_pairs"][agent_pair]["gamma"]

    # Metrics to record
    all_R1 = []
    all_R2 = []
    all_TFT1 = []
    all_TFT2 = []

    for experiment in results["results"]["seeds"]:
        epochs_R1 = []
        epochs_R2 = []
        epoch_TFT1 = []
        epoch_TFT2 = []

        for epoch in experiment["epoch"]:
            end_policy1 = np.array(epoch["P1"])
            end_policy2 = np.array(epoch["P2"])

            V1, V2, comparison1, comparison2 = get_R_std_compare_for_single_policy_pair(end_policy1, end_policy2, r1,
                                                                                        r2,
                                                                                        gamma, comparison_policy,
                                                                                        tolerance)

            # TFT likeliness percentage across both agents
            epoch_TFT1.append(comparison1)
            epoch_TFT2.append(comparison2)
            epochs_R1.append(V1)
            epochs_R2.append(V2)

        all_R1.append(epochs_R1)
        all_R2.append(epochs_R2)
        all_TFT1.append(epoch_TFT1)
        all_TFT2.append(epoch_TFT2)

    return all_R1, all_R2, all_TFT1, all_TFT2


def viewer_friendly_pair(agent_pair="lola1b_vs_nl", as_list=False):
    substitute = {"lola1": "LOLA-Ex", "lola1b": "LOLAb-Ex", "nl": "NL-Ex"}
    agents = agent_pair.split("_")
    if as_list:
        return [substitute[agents[0]], substitute[agents[2]]]
    return substitute[agents[0]] + " vs " + substitute[agents[2]]


def find_the_similarities(dict1, dict2):
    similarities = {}

    if dict1 == dict2:
        return dict1

    elif isinstance(dict1, list) and isinstance(dict2, list):
        same = True
        for i, item in enumerate(dict1):
            if item != dict2[i]:
                same = False
                break
        if same:
            return dict1

    elif isinstance(dict1, dict) and isinstance(dict2, dict):
        for k, v in dict1.items():
            if k in dict2:
                similar = find_the_similarities(v, dict2[k])
                if similar != {}:
                    similarities[k] = similar

    return similarities


def find_the_differences(dict1, dict2):
    if dict1 == dict2:
        return [{}, {}]

    elif isinstance(dict1, list) and isinstance(dict2, list):
        same = True
        for i, item in enumerate(dict1):
            if item != dict2[i]:
                same = False
                break
        if same:
            return [{}, {}]
        else:
            return [dict1, dict2]

    elif isinstance(dict1, dict) and isinstance(dict2, dict):
        sub_dict1 = {}
        sub_dict2 = {}
        for k in set(list(dict1.keys()) + list(dict2.keys())):
            if k in dict1 and k in dict2:
                diffs = find_the_differences(dict1[k], dict2[k])
                if diffs[0] != {}:
                    sub_dict1[k] = diffs[0]
                if diffs[1] != {}:
                    sub_dict2[k] = diffs[1]
            elif k in dict1:
                sub_dict1[k] = dict1[k]
            elif k in dict2:
                sub_dict2[k] = dict2[k]

        return [sub_dict1, sub_dict2]
    else:
        return [dict1, dict2]


def compress_differences(differences):
    compressed = defaultdict(lambda: [])
    for d in differences:
        for k, v in d.items():
            if v not in compressed[k]:
                compressed[k].append(v)
                if isinstance(v, (int, float)):
                    compressed[k].sort()

    return compressed


def key_specific_similarity(key, dict_list):
    different_key_values = []
    common = []

    for d in dict_list:
        if key in d and d[key] not in different_key_values:
            different_key_values.append(d[key])
            common.append(d)

    for i, k in enumerate(different_key_values):
        for d in dict_list:
            if d[key] == k:
                common[i] = find_the_similarities(common[i], d)

    return common