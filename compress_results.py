from result_collection.helper_func import *


# Compresses the result by getting rid of the data in epoch
def compress_result(folder):
    for filename in find_files(folder, "*.json"):
        results = load_results(filename)
        for experiment in results["results"]["seeds"]:
            if "epoch" in experiment:
                del experiment["epoch"]
        save_results(results, filename)
        del results


# Compresses the result by getting rid of only some of the data in epoch
def compress_result_save_epoch(folder, save_epoch_every=50):
    for filename in find_files(folder, "*.json"):
        results = load_results(filename)
        experiments = []
        for experiment in results["results"]["seeds"]:
            if "epoch" in experiment:
                compressed_epochs = []
                for i, iteration in enumerate(experiment["epoch"]):
                    if i % save_epoch_every == 0:
                        compressed_epochs.append(iteration)
                experiment["epoch"] = compressed_epochs
                experiments.append(experiment)
            else:
                print("No epochs to compress")
        results["results"]["seeds"] = experiments
        save_results(results, filename)
        del results


if __name__ == "__main__":
    # compress_result("results/lola1_random_init_policy_robustness/")
    # compress_result_save_epoch("results/lola1_random_init_policy_robustness/")
    compress_result_save_epoch("results/lola1b_random_init_policy_robustness/")
