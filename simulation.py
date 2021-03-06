import argparse
import json
from myLOLA.LOLA_custom import *

import time
import sys, os
import random
import numpy as np
import copy


def load_config(args_to_sub, path="config.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    if "config" in data:
        data = data["config"]
    if args_to_sub is not None:
        substitute(args_to_sub, data)
    return data


def save_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        json.dump(results, outfile, indent=2)


def subst(base_json, path, json_to_sub):
    if len(path) == 1:
        base_json[path[0]] = json_to_sub
        return base_json

    elif path[0] not in base_json:
        sys.stderr.write("{0} not in base_json, creating it anyway\n".format(path[0]))
        acc = base_json
        for p in path[:-1]:
            acc[p] = {}
            acc = acc[p]

    base_json[path[0]] = subst(base_json[path[0]], path[1::], json_to_sub)
    return base_json


def substitute(to_sub_list, json_input):
    for p in to_sub_list:
        # print(p)
        json_input = subst(json_input, p.split("=")[0].strip().split("."), json.loads(p.split("=")[1]))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_from_init_policy_dist(dist):
    # return 0.5
    if dist["name"] == "uniform":
        return np.random.uniform(dist["params"][0], dist["params"][1])
    if dist["name"] == "normal":
        return sigmoid(dist["params"][0] + dist["params"][1] * np.random.randn())
    raise Exception("Random policy initialisation distribution not implemented: " + dist["name"])

def main(args):
    np.random.seed(random.randint(0, 10000000))

    # print(args.to_sub)
    config = load_config(args.to_sub, path=args.input)
    # config["simulation"]["repeats"] = 30
    # config["simulation"]["length"] = 50
    # config["simulation"]["beta"] = 0.5
    # config["simulation"]["eta"] = 0.5
    # config["simulation"]["agent_pair"] = "lola1b_vs_lola1b"
    # dist = "{" + """"name": "normal", "params": [{0}, {1}]""".format(0, 1) + "}"
    # config["agent_pair"]["init_policy_dist"] = json.loads(dist)
    # config["agent_pair"]["init_policy2"] =

    results = {"config": copy.deepcopy(config), "results": {"seeds": []}}
    game = config["simulation"]["game"]
    agent_pair_name = config["simulation"]["agent_pair"]
    repeats = config["simulation"]["repeats"]

    init_policy1_conf = config["agent_pair"]["init_policy1"][:]
    init_policy2_conf = config["agent_pair"]["init_policy2"][:]

    print("Running:", agent_pair_name, game)
    for j in range(repeats):
        init_policy1 = config["agent_pair"]["init_policy1"]
        init_policy2 = config["agent_pair"]["init_policy2"]

        # Replace None values in the policies with values drawn from distribution
        for i, p in enumerate(init_policy1_conf):
            if p is None:
                init_policy1[i] = draw_from_init_policy_dist(config["agent_pair"]["init_policy_dist"])

        for i, p in enumerate(init_policy2_conf):
            if p is None:
                init_policy2[i] = draw_from_init_policy_dist(config["agent_pair"]["init_policy_dist"])
        print(init_policy1)
        seed = config["simulation"]["seed_start"] + j
        print("\tRun:", j, "Seed:", seed)

        # filter settings for run
        all_settings = {**config["agent_pair"], **config["game"], **config["simulation"]}
        run_settings = {k: v for k, v in all_settings.items() if k in run_signature}
        other_settings = {k: v for k, v in all_settings.items() if k not in run_signature}

        agent_pair_class = globals()[agent_pair_name.upper()]
        agent_pair = agent_pair_class(run_settings, other_settings)

        # Run the simulation timing how long it takes to complete
        start_time = time.time()
        _, _, result = agent_pair.run(seed=seed)
        print("\tRun took: ", time.time() - start_time, "sec")

        results["results"]["seeds"].append(result)

    save_results(results, args.output_folder + agent_pair_name + "_" + game + ".json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate Resource Allocation')
    parser.add_argument("-p", "--to_sub", nargs="+",
                        help='parameters which override the config file, '
                             'put each overriding parameter in separate quotes, '
                             '\'tasks.arrival.distribution={"name" : "poisson", "param" : [0.1] }\'')
    parser.add_argument("-o", "--output_folder", help="output file", default="test_results/")
    parser.add_argument("-i", "--input", help="input config file in Json format",
                        default="config.json")
    args = parser.parse_args(sys.argv[1:])
    main(args)

