
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
    print(latex)


if __name__ == "__main__":
    # table_basic_experiments()
    # basic_experiment_replications_table()
    # basic_experiment_replications_figure()
    # basic_experiment_replications_walk_through_space_figure()
    # plot_sigmoid()
    print_experimental_setup()