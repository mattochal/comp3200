from result_collection.helper_func import *
from result_collection.benchmark_metrics import *

SUBS = {'repeats': lambda x: r'\# of repeats: {0}'.format(x),
        'length': lambda x: r'\# of epochs: {0}'.format(x),
        'delta': lambda x: r'$\delta={0}$'.format(x),
        'eta': lambda x: r'$\eta={0}$'.format(x),
        'beta': lambda x: r'$\beta={0}$'.format(x),
        'init_policy_dist': lambda x: 'Initial policy distribution: {0}({1},{2})'.format(x["name"], x["params"][0], x["params"][1]),
        'gamma': lambda x: r'$\gamma={0}$'.format(x),
        'payoff1': lambda x: 'Payoff for agent 0: {0}'.format(x),
        'payoff2': lambda x: 'Payoff for agent 1: {0}'.format(x),
        'game': lambda x: 'Game: {0}'.format(x),
        'agent_pair': lambda x: 'Agent pair: {0}'.format(viewer_friendly_pair(x))}

SUBS_value_only = \
       {'repeats': lambda x: r'\# of repeats: {0}'.format(x),
        'length': lambda x: r'\# of epochs: {0}'.format(x),
        'delta': lambda x: r'$\delta={0}$'.format(x),
        'eta': lambda x: r'$\eta={0}$'.format(x),
        'beta': lambda x: r'$\beta={0}$'.format(x),
        'init_policy_dist': lambda x: 'Initial policy distribution: {0}({1},{2})'.format(x["name"], x["params"][0], x["params"][1]),
        'gamma': lambda x: r'$\gamma={0}$'.format(x),
        'payoff1': lambda x: 'Payoff 0: {0}'.format(x),
        'payoff2': lambda x: 'Payoff 1: {0}'.format(x),
        'game': lambda x: '{0}'.format(x),
        'agent_pair': lambda x: '{0}'.format(viewer_friendly_pair(x))}

IGNORE = ['__order', 'seed_start', 'rollout_length', 'num_rollout', 'init_policy1', 'init_policy2']


STATES = {"IPD": ["s0", "CC", "CD", "DC", "DD"],
          "ISH": ["s0", "SS", "SH", "HS", "HH"],
          "ISD": ["s0", "SS", "SD", "DS", "DD"],
          "IMP": ["s0", "HH", "HT", "TH", "TT"]}

PROB_STATE_LABLES = {"IPD": "cooperation",
                     "ISH": "stag",
                     "ISD": "stop",
                     "IMP": "heads"}


def pprint_dict(my_dict, order=[[]]):
    my_string = ""
    row_num = 0
    row_exists = False
    for r, row in enumerate(order):
        if row_exists:
            my_string += r" \\ "

        row_exists = False
        value_exists = False
        for c in row:
            if c in my_dict:
                if value_exists:
                    my_string += ",   "
                my_string += SUBS[c](my_dict[c])
                row_exists = True
                value_exists = True
            else:
                value_exists = False

    return my_string


def pprint_dict_list(dict_list, order=[[]]):
    my_string = ""
    row_num = 0

    for r, my_dict in enumerate(dict_list):
        if r != 0:
            my_string += r" \\ "

        value_exists = False
        for c, k in enumerate(order):
            if value_exists:
                my_string += ",  "
            if k in my_dict:
                my_string += SUBS_value_only[k](my_dict[k])
                value_exists = True
            else:
                value_exists = False

    return my_string


def print_experimental_setup(folder="../results/lola_through_ST_space/"):
    agent_pair = "lola1_vs_lola1"  # "lola1b_vs_lola1", "lola1b_vs_lola1b"
    game = "unspecified"
    results = collect_experiment_configs(folder, "*{0}_{1}.json".format(agent_pair, game))

    flat_configs = []
    for filename, X in results.items():
        config = X
        flat_config = {}
        for k, v in config.items():
            for k2, v2 in v.items():
                flat_config[k2] = v2
        flat_configs.append(flat_config)

    common_config = flat_configs[0]
    for config in flat_configs[1:]:
        common_config = find_the_similarities(common_config, config)

    differences = []
    for config in flat_configs:
        differences.append(find_the_differences(config, common_config)[0])

    order_common = [['game', 'payoff1', 'payoff2'],
                    ['agent_pair'],
                    ['repeats', 'length'],
                    ['delta', 'eta', 'beta', 'gamma'],
                    ['init_policy_dist']]

    order_game = ['game', 'payoff1', 'payoff2', 'gamma']
    order_agent = ['agent_pair']

    common_setup = pprint_dict(common_config, order=order_common)
    game_setup = pprint_dict_list(key_specific_similarity("game", differences), order=order_game)
    pair_setup = pprint_dict_list(key_specific_similarity("agent_pair", differences), order=order_agent)

    latex = r"""\hline
    \multirow{4}{*}{\begin{tabular}[x]{@{}c@{}}
    Table~\ref{table:replication:IPDIMP:my1} \\
    Figure~\ref{fig:replication-IPD:my_IPD} \\
    Figure~\ref{fig:replication-IMP:my_IMP}
    \end{tabular}}
    & \textbf{Common}
    & \multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}
    """ + common_setup + r"""
    \end{tabular}} \\ \cline{2-3}

    & \textbf{Games}
    & \multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}
    """ + game_setup + r"""
    \end{tabular}} \\ \cline{2-3}

    & \textbf{Pairs}
    & \multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}
    """ + pair_setup + r"""
    \end{tabular}} \\ \cline{2-3}

    & \textbf{Comments}
    & \multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}
    No Comment
    \end{tabular}} \\ \hline
    """
    print(latex)


def table_basic_experiments(folder="../results/basic_experiments/"):
    results = None  # collect_experiment_results(folder, "*.json") correct commented term later

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

        av_R1, std_R1, av_R2, std_R2, av_compare_1, av_compare_2 = None
        #     get_av_end_R_std_TFT(X, COMPARISON[game], TOLERANCE[game], joint=True)

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


def table_delta_eta_results(folder="../results/lola_robust_delta_eta/"):
    game_order = ["IPD"]
    game = game_order[0]
    agent_pair_order = ["lola1_vs_lola1"]
    results = collect_experiment_results(folder, "*{0}.json".format(game), top=10)

    ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0][:1]

    ordered_results = [[None] * len(DELTA) for _ in range(len(ETA))]
    # titles = [None] * len(DELTA)
    for filename, X in results.items():
        delta = X["config"]["agent_pair"]["delta"]
        eta = X["config"]["agent_pair"]["eta"]
        gamma = X["config"]["agent_pair"]["gamma"]
        r1 = X["config"]["game"]["payoff1"]
        r2 = X["config"]["game"]["payoff2"]
        delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
        if delta in DELTA and eta in ETA:
            e_idx = int(delta_eta[1][1:])
            d_idx = int(delta_eta[0][1:])
            epoch_policies = get_epoch_policies(X)
            ordered_results[e_idx][d_idx] = get_av_metrics_over_repeates_for_table(epoch_policies, gamma, r1, r2, game)

    print(TABLE_METRIC_ORDER)
    arr = np.array(ordered_results)
    print(np.shape(arr))
    a = np.reshape(arr, (arr.shape[0], -1))
    print(np.shape(a))
    np.savetxt("foo.csv", a, delimiter="\t")

    # csv = ""
    # for r, row_result in enumerate(ordered_results):
    #     if r != 0:
    #         csv += "\n "
    #
    #     for g, game_result in enumerate(pair_result):
    #         av_R1 = table[p][g][0][0]
    #         std_R1 = table[p][g][0][1]
    #         av_compare_1 = table[p][g][0][2]
    #         csv += ", {0:0.2f}({1:0.2f}), {2:0.1f}".format(av_R1, std_R1, av_compare_1)
    #
    # top_row = ""
    # for game in game_order:
    #     top_row += " , " + game + ", "
    # top_row += ""
    # csv = top_row + csv
    # print(csv.replace(", ", "\t"))

            # titles[d_idx] = r"$\eta={0}$".format(eta)

    # plot_v_timelines_for_delta_eta_combined_plot(ordered_results, STATES[game], titles=titles,
    #                                              prob_state=PROB_STATE_LABLES[game], show=True,
    #                                              filename=folder + "lola_{0}_robust_delta_eta_R".format(game))
"""
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["lola1_vs_lola1"]
    game_order = ["IPD"]

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
"""


def table_st_space_R(folder="../results/lola_uniform_random_init_policy/"):
    agent_pair_order = ["lola1_vs_lola1"]
    game = "IPD"

    randomness = np.linspace(0, 0.5, 51)
    ordered_results = [None] * len(randomness)

    file = folder + "compressed.npy"

    results = collect_experiment_results(folder, "*{0}.json".format(game), top=1)
    my_X = list(results.values())[0]

    if not os.path.exists(file):
        results = collect_experiment_ith_policies(folder, 500-1, "*{0}.json".format(game))

        for filename, X in results.items():
            r = int(filename.split(folder)[1].split("/")[0][1:])
            ordered_results[r] = X

        ordered_results = np.array(ordered_results)
        np.save(file, ordered_results)
    else:
        ordered_results = np.load(file)

    ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0][:1]

    ordered_results = [[None] * len(DELTA) for _ in range(len(ETA))]
    # titles = [None] * len(DELTA)
    for filename, X in results.items():
        delta = X["config"]["agent_pair"]["delta"]
        eta = X["config"]["agent_pair"]["eta"]
        gamma = X["config"]["agent_pair"]["gamma"]
        r1 = X["config"]["game"]["payoff1"]
        r2 = X["config"]["game"]["payoff2"]
        delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
        if delta in DELTA and eta in ETA:
            e_idx = int(delta_eta[1][1:])
            d_idx = int(delta_eta[0][1:])
            epoch_policies = get_epoch_policies(X)
            ordered_results[e_idx][d_idx] = get_av_metrics_over_repeates_for_table(epoch_policies, gamma, r1, r2, game)

    print(TABLE_METRIC_ORDER)
    arr = np.array(ordered_results)
    print(np.shape(arr))
    a = np.reshape(arr, (arr.shape[0], -1))
    print(np.shape(a))
    np.savetxt("foo.csv", a, delimiter="\t")


if __name__ == "__main__":
    # table_basic_experiments()
    print_experimental_setup()
    # table_delta_eta_results()