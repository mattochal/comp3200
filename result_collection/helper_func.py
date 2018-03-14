import numpy as np
import os
import fnmatch
import json


# Given directory to look in
# and a pattern in regex for file ending
# find the files matching files
# You can choose to ignore matches at the root of the specified directory
def find_files(directory, pattern, ignore_root=True):
    for root, dirs, files in os.walk(directory):
        if not (root == directory and ignore_root):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename


# Loads json results given path to json file
def load_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# Saves result to given filename
def save_results(results, filename):
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
def collect_experiment_end_policies(folder, pattern='*.json', top=None):
    results = dict()
    filenames = []
    for filename in find_files(folder, pattern):
        filenames.append(filename)

    for filename in sorted(filenames)[:top]:
        print("Loading: ", filename)
        X = get_end_policies(load_results(filename))
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
        print("Loading: ", filename)
        X = get_ith_policies(load_results(filename), i)
        results[filename] = X
    return results


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
def get_av_end_R_std_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]):
    av_R_1, av_R_2, av_compare_1, av_compare_2 = get_end_R_std_TFT(results, comparison_policy)

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
def get_end_R_std_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]):

    # Get some info about the simulation experiment
    game = results["config"]["simulation"]["game"]
    agent_pair = results["config"]["simulation"]["agent_pair"]
    r1 = np.array(results["config"]["games"][game]["payoff1"])
    r2 = np.array(results["config"]["games"][game]["payoff2"])
    gamma = results["config"]["agent_pairs"][agent_pair]["gamma"]

    # Metrics to record
    all_R_1 = []
    all_R_2 = []
    all_compare_1 = []
    all_compare_2 = []

    for experiment in results["results"]["seeds"]:
        end_policy = np.array([experiment["P1"], experiment["P2"]])

        # Calculated as an absolute difference between end policy and
        comparison = 1 - np.mean(np.abs(end_policy - comparison_policy), 1)

        # TFT likeliness percentage across both agents
        all_compare_1.append(comparison[0])
        all_compare_2.append(comparison[1])

        V1, V2 = calculate_value_fn_from_policy(end_policy, r1, r2, gamma)
        all_R_1.append(V1 * (1 - gamma))
        all_R_2.append(V2 * (1 - gamma))

    return all_R_1, all_R_2, all_compare_1, all_compare_2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# over the repeat through all the iterations in each epoch
# and return the similarity to a given comparison policy (assumed TFT)
def get_epoch_R_std_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]):

    # Get some info about the simulation experiment
    game = results["config"]["simulation"]["game"]
    agent_pair = results["config"]["simulation"]["agent_pair"]
    r1 = np.array(results["config"]["games"][game]["payoff1"])
    r2 = np.array(results["config"]["games"][game]["payoff2"])
    gamma = results["config"]["agent_pairs"][agent_pair]["gamma"]

    # Metrics to record
    all_av_R_1 = []
    all_av_R_2 = []
    all_av_compare_1 = []
    all_av_compare_2 = []

    for experiment in results["results"]["seeds"]:
        av_R_1 = []
        av_R_2 = []
        av_compare_1 = []
        av_compare_2 = []

        for epoch in experiment["epoch"]:
            end_policy = np.array([epoch["P1"], epoch["P2"]])

            # Calculated as an absolute difference between end policy and
            comparison = 1 - np.mean(np.abs(end_policy - comparison_policy), 1)

            # TFT likeliness percentage across both agents
            av_compare_1.append(comparison[0])
            av_compare_2.append(comparison[1])

            V1, V2 = calculate_value_fn_from_policy(end_policy, r1, r2, gamma)
            av_R_1.append(V1 * (1 - gamma))
            av_R_2.append(V2 * (1 - gamma))

        all_av_R_1.append(av_R_1)
        all_av_R_2.append(av_R_2)
        all_av_compare_1.append(av_compare_1)
        all_av_compare_2.append(av_compare_2)

    return all_av_R_1, all_av_R_2, all_av_compare_1, all_av_compare_2


# Given a single experiment result in json format
# return the average reward R per time step and standard deviation
# over the repeat through all the iterations in each epoch
# and calculate and return an overall average for those values
def get_av_epoch_R_std_TFT(results, comparison_policy=[[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]):
    all_av_R_1, all_av_R_2, all_av_compare_1, all_av_compare_2 =\
        get_epoch_R_std_TFT(results, comparison_policy)

    all_std_av_reward_1 = np.std(all_av_R_1, axis=0)
    all_av_R_1 = np.mean(all_av_R_1, axis=0)

    all_std_av_reward_2 = np.std(all_av_R_2, axis=0)
    all_av_R_2 = np.mean(all_av_R_2, axis=0)

    all_av_compare_1 = np.mean(all_av_compare_1, axis=0)
    all_av_compare_2 = np.mean(all_av_compare_2, axis=0)

    return all_av_R_1, all_std_av_reward_1, all_av_R_2, all_std_av_reward_2, all_av_compare_1, all_av_compare_2