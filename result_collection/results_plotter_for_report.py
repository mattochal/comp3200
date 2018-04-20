from result_collection.collect_batch_results import *
from result_collection.helper_func import collect_experiment_results
from result_collection.results_plotter import *

STATES = {"IPD": ["s0", "CC", "CD", "DC", "DD"],
          "ISH": ["s0", "SS", "SH", "HS", "HH"],
          "ISD": ["s0", "SS", "SD", "DS", "DD"],
          "IMP": ["s0", "HH", "HT", "TH", "TT"]}

PROB_STATE_LABLES = {"IPD": "cooperation",
                     "ISH": "stag",
                     "ISD": "stop",
                     "IMP": "heads"}

COMPARISON = {"IPD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
              "ISH": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
              "ISD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
              "IMP": [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]]}

TOLERANCE = {"IPD": 0.5,
             "ISH": 0.5,
             "ISD": 0.5,
             "IMP": 0.25}


def table_basic_experiments(folder="../results/basic_experiments/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1", "lola1_vs_nl",
                        "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    game_order = ["IPD", "ISH", "ISD", "IMP"]

    # storing [R std %compare] for two players, for all pairs and games
    table = np.zeros((len(agent_pair_order), len(game_order), 2, 3)) * 0.0

    for filename, X in results.items():
        game = filename.split("../results/basic_experiments/")[1].split("/")[0]
        pair = filename.split("../results/basic_experiments/")[1].split("/")[1]
        game_idx = game_order.index(game)
        pair_idx = agent_pair_order.index(pair)

        av_R1, std_R1, av_R2, std_R2, av_compare_1, av_compare_2 = \
            get_av_end_R_std_TFT(X, COMPARISON[game], TOLERANCE[game], joint=True)

        for i, v in enumerate([av_R1, std_R1, av_compare_1]):
            table[pair_idx][game_idx][0][i] = v

        for i, v in enumerate([av_R2, std_R2, av_compare_2]):
            table[pair_idx][game_idx][1][i] = v

    # Player 1 only table
    csv = ""
    for p, pair_result in enumerate(table):
        csv += "\n" + viewer_friendly_pair(agent_pair_order[p])
        for g, game_result in enumerate(pair_result):
            av_R1 = table[p][g][0][0]
            std_R1 = table[p][g][0][1]
            av_compare_1 = table[p][g][0][2]
            csv += ", {0:0.2f}({1:0.2f}), {2:0.1f}".format(av_R1, std_R1, av_compare_1)

    top_row = ""
    for game in game_order:
        top_row += " , " + game + ", "
    top_row += ""
    csv = top_row + csv
    print(csv.replace(", ", "\t"))


def basic_experiment_replications_table(folder="../results/basic_lola_replication_50_epochs/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1"]
    #                     , "lola1_vs_nl", "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    game_order = ["IPD", "IMP"]  # "ISH", "ISD",

    # storing [R std %compare] for two players, for all pairs and games
    table = np.zeros((len(agent_pair_order), len(game_order), 2, 3)) * 0.0

    for filename, X in results.items():
        game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        print(game, pair)
        if game in game_order and pair in agent_pair_order:
            game_idx = game_order.index(game)
            pair_idx = agent_pair_order.index(pair)

            av_R1, std_R1, av_R2, std_R2, av_compare_1, av_compare_2 = \
                get_av_end_R_std_TFT(X, COMPARISON[game], TOLERANCE[game], joint=True)

            for i, v in enumerate([av_R1, std_R1, av_compare_1]):
                table[pair_idx][game_idx][0][i] = v

            for i, v in enumerate([av_R2, std_R2, av_compare_2]):
                table[pair_idx][game_idx][1][i] = v

    # Player 1 only table
    csv = ""
    for p, pair_result in enumerate(table):
        csv += "\n" + viewer_friendly_pair(agent_pair_order[p])
        for g, game_result in enumerate(pair_result):
            arg = 0  # np.argmax([table[p][g][i][2] for i in [0, 1]])
            av_R1 = table[p][g][arg][0]
            std_R1 = table[p][g][arg][1]
            av_compare_1 = table[p][g][arg][2]
            csv += ", {2:0.1f}, {0:0.2f}({1:0.2f})".format(av_R1, std_R1, av_compare_1 * 100)

    top_row = ""
    for game in game_order:
        top_row += " , " + game + ", "
    top_row += ""
    csv = top_row + csv
    print(csv.replace(", ", " & "))


def basic_experiment_replications_figure(folder="../results/basic_lola_replication_200_epochs/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1"]
    #                     , "lola1_vs_nl", "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    game_order = ["IPD", "IMP"]  # "ISH", "ISD",

    ordered_results = [[None]*len(game_order) for _ in range(len(agent_pair_order))]
    for filename, X in results.items():
        game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        print(game, pair)
        if game in game_order and pair in agent_pair_order:
            game_idx = game_order.index(game)
            pair_idx = agent_pair_order.index(pair)
            ordered_results[game_idx][pair_idx] = X

    for g, game in enumerate(game_order):
        plot_policies_and_v_timeline(ordered_results[g], STATES[game], filename_to_save=folder + "my_" + game + ".pdf",
                                     prob_state=PROB_STATE_LABLES[game])


def basic_experiment_replications_walk_through_space_figure(folder="../results/basic_lola_replication_200_epochs/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1"]
    game_order = ["IPD", "IMP"]

    intervals = [[0, 4], [4, 9], [9, 49], [49, 99], [99, 199]]
    # intervals = [[0, i] for i in [3, 10, 50, 100, 199]]

    ordered_results = [[None] * len(game_order) for _ in range(len(agent_pair_order))]
    for filename, X in results.items():
        game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        print(game, pair)
        if game in game_order and pair in agent_pair_order:
            game_idx = game_order.index(game)
            pair_idx = agent_pair_order.index(pair)
            ordered_results[game_idx][pair_idx] = X

    for p, pair in enumerate(agent_pair_order):
        for g, game in enumerate(game_order):
            plot_policy_walk_through_space(ordered_results[g][p], intervals, STATES[game], prob_state=PROB_STATE_LABLES[game],
                                         filename_to_save=folder + "my_" + pair + "_" + game + "_space_walk_5.pdf", top=[None, 5])


def plot_sigmoid():
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.linspace(-5, 5, 41)
    y = sigmoid(x)

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.plot(x, y)
    ax.set_xlabel(r'$\theta^s$')
    ax.set_ylabel(r'$P(C|s)$')
    plt.grid(True)
    plt.subplots_adjust(left=0.17, right=0.95, top=0.88, bottom=0.23, wspace=0.1, hspace=0.27)
    # plt.title(r'Sigmoid curve over parameters $\theta^s$')
    # plt.show()
    plt.savefig("sigmoid.pdf")


def lola_robust_delta_eta(folder="../results/lola_robust_delta_eta/"):
    game_order = ["IPD"]
    game = game_order[0]
    agent_pair_order = ["lola1_vs_lola1"]

    results = collect_experiment_results(folder, "*{0}.json".format(game))

    ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    # ETA = np.linspace(0.1, 2.0, 10)#[:4]
    # DELTA = np.linspace(0.1, 2.0, 10)#[:2]

    ordered_results = [[None] * len(DELTA) for _ in range(len(ETA))]
    titles = [None] * len(DELTA)
    for filename, X in results.items():
        delta = X["config"]["agent_pair"]["delta"]
        eta = X["config"]["agent_pair"]["eta"]
        delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
        if delta in DELTA and eta in ETA:
            e_idx = int(delta_eta[1][1:])
            d_idx = int(delta_eta[0][1:])
            ordered_results[e_idx][d_idx] = X
            titles[d_idx] = r"$\eta={0}$".format(eta)

    plot_v_timelines_for_delta_eta_combined_plot(ordered_results, STATES[game], titles=titles,
                                                  prob_state=PROB_STATE_LABLES[game], show=True,
                                                  filename=folder + "lola_{0}_robust_delta_eta_R".format(game))


def lola_robust_delta_eta_policies_grid(folder="../results/lola_robust_delta_eta/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["lola1_vs_lola1"]
    game_order = ["IPD"]

    ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0]

    ordered_results = [[None] * len(ETA) for _ in range(len(DELTA))]
    for filename, X in results.items():
        delta = X["config"]["agent_pair"]["delta"]
        eta = X["config"]["agent_pair"]["eta"]
        delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
        if delta in DELTA and eta in ETA:
            e_idx = int(delta_eta[1][1:])
            d_idx = int(delta_eta[0][1:])
            ordered_results[d_idx][e_idx] = get_end_policies(X)
            # titles[d_idx] = r"$\detla={0}$".format(delta)

    game = game_order[0]
    keys = [[r"$\delta={0:.2f}, \eta={1:.2f}$".format(d, e) for e in ETA] for d in DELTA]
    plot_2ax_policies(np.array(ordered_results), keys, "",
                      filename=folder + "lola_{0}_robust_delta_eta_policies_grid.pdf".format(game),
                      show=False, figsize=(30, 30), prob_state=PROB_STATE_LABLES[game], states=STATES[game])


def lola_robust_delta_eta_R_conv_tft2_graph(folder="../results/lola_robust_delta_eta/"):
    ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    agent_pair_order = ["lola1_vs_lola1"]
    game = "IPD"

    delta = None
    eta = 1

    if delta is not None:
        d_idx = DELTA.index(delta)
        pattern = "*D0" + str(d_idx) + "*"+game+".json"
        ordered_results = [None] * len(ETA)
    else:
        e_idx = ETA.index(eta)
        pattern = "*E0" + str(e_idx) + "*"+game+".json"
        ordered_results = [None] * len(DELTA)

    results = collect_experiment_results(folder, pattern)

    for filename, X in results.items():
        d = X["config"]["agent_pair"]["delta"]
        e = X["config"]["agent_pair"]["eta"]
        delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
        if d in DELTA and e in ETA:
            e_idx = int(delta_eta[1][1:])
            d_idx = int(delta_eta[0][1:])
            if delta is not None:
                ordered_results[e_idx] = X
            else:
                ordered_results[d_idx] = X

    plot_delta_eta_row_col_benchmarks(ordered_results, show=False, delta_or_eta="eta" if delta is not None else "delta",
                                      filename=folder + "lola_{0}_robust_delta_eta_graph_e100.pdf".format(game))


def lola_robust_to_random_policy_init(folder="../results/lola1_random_init_policy_robustness/"):
    game = "IPD"
    results = collect_experiment_results(folder, "*{0}.json".format(game))

    randomness = np.linspace(0, 0.5, 51)
    sorted_results = [None for _ in randomness]

    ith = 1000

    def index(filename):
        return int(re.findall('R\d+', filename)[0][1:])

    for filename, X in results.items():
        i = index(filename)
        if game in filename and i < len(sorted_results):
            sorted_results[i] = get_ith_policies(results, ith)



if __name__ == "__main__":
    # table_basic_experiments()
    # basic_experiment_replications_table()
    # basic_experiment_replications_figure()
    # basic_experiment_replications_walk_through_space_figure()
    # plot_sigmoid()
    # lola_robust_delta_eta()
    lola_robust_delta_eta_policies_grid()
    # lola_robust_delta_eta_R_conv_tft2_graph()
