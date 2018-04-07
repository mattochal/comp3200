from result_collection.collect_batch_results import *
from result_collection.helper_func import *


def table_basic_experiments(folder="results/basic_experiments/"):
    results = collect_experiment_end_R_std_TFT(folder, "*.json")

    keys = ["After training for {0} epochs".format(r * 50) for r in np.linspace(0, 20, 21)]
    agent_pair_order = [ "nl_vs_nl", "lola1_vs_lola1", "lola1_vs_nl",
                         "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    game_order = ["IPD", "ISH", "ISD", "IMP"]

    comparison_policies = {"IPD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "ISH": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "ISD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "IMP": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]}

    tolerances = {"IPD": 0.05,
                  "ISH": 0.05,
                  "ISD": 0.05,
                  "IMP": 0.05}

    def index(filename):
        return int(re.findall('R\d+', filename)[0][1:])
    #
    # for filename, X in results.items():
    #     i = index(filename)
    #     if game in filename and i < len(sorted_results):
    #         sorted_results[i] = np.array(X)
    #
    # plot_1ax_R_std_TFT_through_epochs(np.array(sorted_results), keys,
    #                                   "How randomness in policy parameter initialisation"
    #                                   " affects the end policy of the {1} agents in the "
    #                                   "{0} game".format(game, agents),
    #                                   show=False, figsize=(15, 40), filename=folder[:-1] + "_through_epochs")
    #

if __name__ == "__main__":
    # Table for including LOLA vs LOLA and NL vs NL in IMP and IPD
    # TODO: run for 1000 repeats, with IPD of length 200
    pass
