import argparse
import json
from LOLA_pytorch.LOLA_vs_LOLA import LOLA_VS_LOLA
from LOLA_pytorch.LOLAOM_vs_LOLAOM import LOLAOM_VS_LOLAOM
from LOLA_pytorch_complete.LOLA_custom import LOLA1_VS_LOLA1, LOLA1B_VS_LOLA1B

import time
import sys, os
import random
import numpy as np


def load_config(args_to_sub, path="config.json"):
    with open(path, 'r') as f:
        data = json.load(f)
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
    else:
        base_json[path[0]] = subst(base_json[path[0]], path[1::], json_to_sub)
        return base_json


def substitute(to_sub_list, json_input):
    for p in to_sub_list:
        json_input = subst(json_input, p.split("=")[0].strip().split("."), json.loads(p.split("=")[1]))


def draw_from_init_policy_dist(dist):
    if dist["name"] == "uniform":
        return np.random.uniform(dist["params"][0], dist["params"][1])


def main(args):
    config = load_config(args.to_sub)
    results = {"config": config, "results": {"seeds": []}}
    game = config["simulation"]["game"]
    agent_pair_name = config["simulation"]["agent_pair"]

    np.random.seed(random.randint(0, 10000000))
    init_policy1_conf = config["games"][game]["init_policy1"][:]
    init_policy2_conf = config["games"][game]["init_policy2"][:]

    print("Running:", agent_pair_name, game)
    for j in range(config["simulation"]["repeats"]):
        init_policy1 = config["games"][game]["init_policy1"]
        init_policy2 = config["games"][game]["init_policy2"]

        for i, p in enumerate(init_policy1_conf):
            if p is None:
                init_policy1[i] = draw_from_init_policy_dist(config["simulation"]["random_init_policy_dist"])

        for i, p in enumerate(init_policy2_conf):
            if p is None:
                init_policy2[i] = draw_from_init_policy_dist(config["simulation"]["random_init_policy_dist"])

        start_time = time.time()
        seed = config["simulation"]["seed_start"] + j
        print("\tRun:", j, "Seed:", seed)

        agent_pair_class = globals()[agent_pair_name.upper()]
        agent_pair = agent_pair_class(config["agent_pairs"][agent_pair_name], config["games"][game])

        _, _, result = agent_pair.run(seed=seed)
        results["results"]["seeds"].append(result)
        print("\tRun took: ", time.time() - start_time, "sec")

    config["games"][game]["init_policy1"] = init_policy1_conf
    config["games"][game]["init_policy2"] = init_policy1_conf

    save_results(results, args.output_folder + agent_pair_name + "_" + game + ".json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate Resource Allocation')
    parser.add_argument("-p", "--to_sub", nargs="+",
                        help='parameters which override the config file, '
                             'put each overriding parameter in separate quotes, '
                             '\'tasks.arrival.distribution={"name" : "poisson", "param" : [0.1] }\'')
    parser.add_argument("-o", "--output_folder", help="output file", default="results/")
    parser.add_argument("-i", "--input", help="input config file in Json format", default="config.json")
    args = parser.parse_args(sys.argv[1:])
    # print(sys.argv)
    main(args)

