import argparse
import json
from LOLA_pytorch.LOLA_vs_LOLA import LOLA_VS_LOLA
from LOLA_pytorch.LOLAOM_vs_LOLAOM import LOLAOM_VS_LOLAOM
import time


def load_config(args_to_sub, path="config.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    if args_to_sub is not None:
        substitute(args_to_sub, data)
    return data


def save_results(results, filename):
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


def main(args):
    config = load_config(args.to_sub)
    results = {"config": config, "results": {"seeds": []}}
    game = config["simulation"]["game"]
    agent_pair_name = config["simulation"]["agent_pair"]

    print("Running:", agent_pair_name, game)
    for i in range(config["simulation"]["repeats"]):
        start_time = time.time()
        seed = config["simulation"]["seed_start"] + i
        print("\tRun:", i, "Seed:", seed)

        agent_pair_class = globals()[agent_pair_name.upper()]
        agent_pair = agent_pair_class(config["agent pairs"][agent_pair_name], config["games"][game])
        _, _, result = agent_pair.run(seed=seed)
        results["results"]["seeds"].append(result)
        print("\tRun took: ", time.time()-start_time, "ms")

    save_results(results, "result_" + agent_pair_name + "_" + game + "_100.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate Resource Allocation')
    parser.add_argument("-p", "--to_sub", nargs="+",
                        help='parameters which override the config file, '
                             'put each overriding parameter in separate quotes, '
                             '\'tasks.arrival.distribution={ "name" : "poisson", "param" : [0.1] }\'')
    args = parser.parse_args()
    main(args)