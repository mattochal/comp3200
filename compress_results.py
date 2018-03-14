from result_collection.helper_func import *


# Compresses the result by getting rid of the data in epoch
def compress_result(folder):
    for filename in find_files(folder, "*.json"):
        print(filename)
        results = load_results(filename)
        for experiment in results["results"]["seeds"]:
            if "epoch" in experiment:
                del experiment["epoch"]
        save_results(results, filename)
        del results


if __name__ == "__main__":
    compress_result("results/lola1_random_init_policy_robustness/")
