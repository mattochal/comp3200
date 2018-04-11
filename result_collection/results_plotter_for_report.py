from result_collection.collect_batch_results import *
from result_collection.helper_func import *
from result_collection.results_plotter import *



def table_basic_experiments(folder="../results/basic_experiments/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1", "lola1_vs_nl",
                        "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    game_order = ["IPD", "ISH", "ISD", "IMP"]

    comparison_policies = {"IPD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "ISH": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "ISD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "IMP": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]]}

    tolerances = {"IPD": 0.5,
                  "ISH": 0.5,
                  "ISD": 0.5,
                  "IMP": 0.5}

    # storing [R std %compare] for two players, for all pairs and games
    table = np.zeros((len(agent_pair_order), len(game_order), 2, 3)) * 0.0

    for filename, X in results.items():
        game = filename.split("../results/basic_experiments/")[1].split("/")[0]
        pair = filename.split("../results/basic_experiments/")[1].split("/")[1]
        game_idx = game_order.index(game)
        pair_idx = agent_pair_order.index(pair)

        av_R1, std_R1, av_R2, std_R2, av_compare_1, av_compare_2 = \
            get_av_end_R_std_TFT(X, comparison_policies[game], tolerances[game], joint=True)

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

    comparison_policies = {"IPD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "ISH": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "ISD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
                           "IMP": [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]]}

    tolerances = {"IPD": 0.5,
                  "ISH": 0.5,
                  "ISD": 0.5,
                  "IMP": 0.25}

    states = {"IPD": ["s0", "CC", "CD", "DC", "DD"],
              "ISH": ["s0", "SS", "SH", "HS", "HH"],
              "ISD": ["s0", "SS", "SD", "DS", "DD"],
              "IMP": ["s0", "HH", "HT", "TH", "TT"]}

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
                get_av_end_R_std_TFT(X, comparison_policies[game], tolerances[game], joint=True)

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

    # comparison_policies = {"IPD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
    #                        "ISH": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
    #                        "ISD": [[1, 1, 0, 1, 0], [1, 1, 1, 0, 0]],
    #                        "IMP": [[0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]]}
    #
    # tolerances = {"IPD": 0.5,
    #               "ISH": 0.5,
    #               "ISD": 0.5,
    #               "IMP": 0.25}

    states = {"IPD": ["s0", "CC", "CD", "DC", "DD"],
              "ISH": ["s0", "SS", "SH", "HS", "HH"],
              "ISD": ["s0", "SS", "SD", "DS", "DD"],
              "IMP": ["s0", "HH", "HT", "TH", "TT"]}

    prob_states = {"IPD": "cooperation",
                   "ISH": "stag",
                   "ISD": "stop",
                   "IMP": "heads"}

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
        plot_policies_and_v_timeline(ordered_results[g], states[game], filename_to_save=folder + "my_" + game + ".pdf",
                                     prob_state=prob_states[game])


def basic_experiment_replications_walk_through_space_figure(folder="../results/basic_lola_replication_200_epochs/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1"]
    game_order = ["IPD", "IMP"]
    states = {"IPD": ["s0", "CC", "CD", "DC", "DD"], "IMP": ["s0", "HH", "HT", "TH", "TT"]}
    prob_states = {"IPD": "cooperation", "IMP": "heads"}

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
            plot_policy_walk_through_space(ordered_results[g][p], intervals, states[game], prob_state=prob_states[game],
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

def print_experimental_setup():
    folder = "../results/basic_lola_replication_50_epochs/"
    results = collect_experiment_results(folder, "*.json")

    flat_configs = []
    for filename, X in results.items():
        config = X["config"]
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

    # latex = r"""\multirow{4}{*}{Table 2} & """ \
    #         + r"""\textbf{Common setup} & """\
    #         + r"""\multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}""" + common_setup \
    #         + r"""\end{tabular}} \\ \cline{2-3} """
    # latex += "\n" + r""" & """ \
    #          + r"""\textbf{Games} & """ \
    #          + r"""\multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}""" + game_setup \
    #          + r"""\end{tabular}} \\ \cline{2-3} """
    # latex += "\n" + r""" & """ \
    #          + r"""\textbf{Agent Pairs} & """ \
    #          + r"""\multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}""" + pair_setup \
    #          + r"""\end{tabular}} \\ \cline{2-3} """
    # latex += "\n" + r""" & """ \
    #          + r"""\textbf{Comments} & """ \
    #          + r"""\multicolumn{1}{l|}{ \begin{tabular}[c]{@{}l@{}}""" + "No Comment" \
    #          + r"""\end{tabular}} \\ \hline """

    # latex += "\n" + r""" & \textbf{Agent Pairs} & \begin{tabular}[c]{@{}l@{}} \multicolumn{1}{l|}{""" + pair_setup + r"""} \end{tabular} \\ \cline{2-3}"""
    # latex += "\n" + r""" & \textbf{Comments} & \begin{tabular}[c]{@{}l@{}} \multicolumn{1}{l|}{""" + "No Comment" + r"""} \end{tabular} \\ \hline"""

    print(latex)
    # print(common_config)
    # print(key_specific_similarity("game", differences))
    # print(key_specific_similarity("agent_pair", differences))


if __name__ == "__main__":
    # table_basic_experiments()
    # basic_experiment_replications_table()
    # basic_experiment_replications_figure()
    # basic_experiment_replications_walk_through_space_figure()
    # plot_sigmoid()
    print_experimental_setup()