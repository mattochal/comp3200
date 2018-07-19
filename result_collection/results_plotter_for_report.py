from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from result_collection.collect_batch_results import *
from result_collection.helper_func import collect_experiment_results
from result_collection.results_plotter import *

STATES = {"IPD": ["s0", "CC", "CD", "DC", "DD"],
          "ISH": ["s0", "SS", "SH", "HS", "HH"],
          "ISD": ["s0", "SS", "SD", "DS", "DD"],
          "IMP": ["s0", "HH", "HT", "TH", "TT"]}

PROB_STATE_LABLES = {"IPD": "C",
                     "ISH": "stag",
                     "ISD": "stop",
                     "IMP": "H"}

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
    settings = [dict(), dict()]

    ordered_results = [[None]*len(game_order) for _ in range(len(agent_pair_order))]
    for filename, X in results.items():
        game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        print(game, pair)
        if game in game_order and pair in agent_pair_order:
            game_idx = game_order.index(game)
            pair_idx = agent_pair_order.index(pair)
            ordered_results[game_idx][pair_idx] = X
            settings[game_idx] = dict(gamma=X["config"]["agent_pair"]["gamma"],
                                      r1=X["config"]["game"]["payoff1"],
                                      r2=X["config"]["game"]["payoff2"])

    for g, game in enumerate(game_order):
        plot_policies_and_v_timeline(ordered_results[g], STATES[game], filename_to_save=folder + "my_" + game + ".pdf",
                                     prob_state=PROB_STATE_LABLES[game], **settings[g])


def basic_experiment_replications_walk_through_space_figure(folder="../results/basic_lola_replication_200_epochs/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1"]
    game_order = ["IPD", "IMP"]

    intervals = [[0, 4], [4, 9], [9, 49], [49, 199]]
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
            plot_policy_walk_through_space(ordered_results[g][p], intervals, STATES[game], prob_state=PROB_STATE_LABLES[game], figsize=(10, 3),
                                           filename_to_save=folder + "my_" + pair + "_" + game + "_space_walk_all.pdf", top=[None, None])


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


def plot_st_space():
    pts = int(101*1.1)
    r = 1
    p = 0
    S = np.linspace(-1-0.1, 1+0.1, pts)
    T = np.linspace(0-0.1, 2+0.1, pts)

    def st():
        lbls = ["Harmony", "Prisoner's Dilemma", "Stag Hunt", "Snow Drift"]
        z = np.zeros((pts, pts))
        for i, s in enumerate(S):
            for j, t in enumerate(T):
                if t > r > p > s:
                    z[i][j] = 1
                if r > t >= p > s:
                    z[i][j] = 2
                if t > r and s >= p:
                    z[i][j] = 3
        return lbls, z

    def normalise(rewards):
        P = rewards[3]
        rewards = rewards - P
        R = rewards[0]
        return (rewards / R)[1:3]

    fig, ax = plt.subplots(figsize=(6, 3))

    n_bin = 4
    cmap_name = 'my_list'
    cols = ["gray", "red", "green", "blue"]
    alpha = 0.55
    cm = LinearSegmentedColormap.from_list(cmap_name, cols, N=n_bin)

    lables, z = st()
    ax.imshow(z, interpolation='nearest', origin='lower', cmap=cm, alpha=alpha, label=lables,
                  extent=[min(T) - 0.5 / pts, max(T) + 0.5 / pts, min(S) - 0.5 / pts, max(S) + 0.5 / pts], vmax=n_bin, vmin=-1)

    PD = normalise(np.array([-1, -3, 0, -2]))
    SH = normalise(np.array([2, 0, 1, 1]))
    SD = normalise(np.array([-1, -2, 0, -3]))
    ax.scatter(PD[1], PD[0], c=cols[1], s=20)
    ax.scatter(SH[1], SH[0], c=cols[2], s=20)
    ax.scatter(SD[1], SD[0], c=cols[3], s=20)

    patches = []
    for c, lbl in enumerate(lables):
        patch = mpatches.Patch(color=cols[c], label=lbl, alpha=alpha)
        patches.append(patch)

    ax.legend(handles=patches, bbox_to_anchor=(2.05, 0.75), shadow=True, ncol=1)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])

    ax.set_xlabel('T')
    ax.set_ylabel('S')

    ax.set_xticks(np.linspace(0, 2, 5))
    ax.set_xlim((0-0.1, 2+0.1))

    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.set_ylim((-1-0.1, 1+0.1))
    # plt.grid(True)
    plt.subplots_adjust(left=0.1, right=0.67, top=0.88, bottom=0.23, wspace=0.1, hspace=0.27)
    plt.savefig("st-space.pdf")
    # plt.show()


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
    # ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    ETA = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0, 15.0, 20.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
    agent_pair_order = ["lola1_vs_lola1"]
    game = "IPD"

    delta = 0.1
    eta = None

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
                                      filename=folder + "lola_{0}_robust_delta_eta_graph_d010.pdf".format(game))


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


def metrics_through_time_graph(folder="../results/basic_lola_replication_200_epochs/"):
    results = collect_experiment_results(folder, "*.json")

    agent_pair_order = ["lola1_vs_lola1"]
    game_order = ["IPD"]

    for filename, X in results.items():
        game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        print(game, pair)
        if game in game_order and pair in agent_pair_order:
            metrics = [(["R", "R (agent 1)"], lambda x1, x2: R(x1, x2, gamma=X["config"]["agent_pair"]["gamma"],
                                      r1=X["config"]["game"]["payoff1"],
                                      r2=X["config"]["game"]["payoff2"]), 0, ["red", "orange"], True),
                       (["%TFT"], lambda x1, x2: tft(x1, x2), 1, ["red"], True),
                       (["%TFT2"], lambda x1, x2: tft2(x1, x2), 1, ["blue"], True),
                       (["E[Y(CC)]"], lambda x1, x2: exp_s(x1, x2, state=1), 2, ["blue"], True),
                       (["E[Y(CD)]"], lambda x1, x2: exp_s(x1, x2, state=2), 2, ["orange"], True),
                       (["E[Y(DC)]"], lambda x1, x2: exp_s(x1, x2, state=3), 2, ["green"], True),
                       (["E[Y(DD)]"], lambda x1, x2: exp_s(x1, x2, state=4), 2, ["red"], True)]
            ylabels = ["Av. reward per step, R", "Av. % of TFT policy", "Expected state visits, E[Y(s)]"]
            yticks = [np.linspace(-2, -1, 6), np.linspace(0, 100, 6), np.linspace(0, 100, 6)]
            ybounds = [(-2 - 0.05, -1 + 0.05), (0 - 5, 100 + 5), (0 - 5, 100 + 5)]
            plot_metrics_timeline(X, metrics, filename= folder + "plot_R_TFT_ECC.pdf", n_metrics=3, show=False,
                                  ylabels=ylabels, yticks=yticks, ybounds=ybounds)


def lola_single_value_policy_init_walk_through_space_figure(folder="../results/lola_single_value_policy_init/"):
    results = collect_experiment_results(folder, "*.json")

    # agent_pair_order = ["lola1_vs_lola1"]
    # game_order = ["IPD"]

    intervals = [[0, 1], [1, 499]]
    init_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ordered_results = [None for _ in range(len(init_values))]
    for filename, X in results.items():
        v = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        print(v)
        val_idx = int(v)
        ordered_results[val_idx] = X

    for v, value in enumerate(init_values):
        plot_policy_walk_through_space(ordered_results[v], intervals, STATES["IPD"], prob_state=PROB_STATE_LABLES["IPD"], figsize=(5, 3),
                                       filename_to_save=folder + "lola_v" + str(v) + "_space_walk_all_2.pdf", top=[None, None])


# def lola_delta_eta_walk_through_space_figure(folder="../results/lola_robust_delta_eta/"):
    # agent_pair = "lola1_vs_lola1"
    # game = "IPD"
    # results = collect_experiment_results(folder, "*"+game+"*.json")
    #
    # DELTA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0]
    # ETA = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0, 5.0]
    #
    # de_pairs = np.array([[0.1, 5.0], [0.5, 0.5], [5.0, 5.0], [2.0, 0.1]])
    #
    # intervals = [[0, 4], [4, 9], [9, 49], [49, 199]]
    # # for pair in de_pairs:
    #
    # # ordered_results = [[None] * len(ETA) for _ in range(len(DELTA))]
    # for filename, X in results.items():
    #     delta = X["config"]["agent_pair"]["delta"]
    #     eta = X["config"]["agent_pair"]["eta"]
    #     delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
    #     if delta in de_pairs[:, 0] and eta in de_pairs[:, 1] and game in filename:
    #         e_idx = int(delta_eta[1][1:])
    #         d_idx = int(delta_eta[0][1:])
    #         plot_policy_walk_through_space(results, intervals, STATES["IPD"],
    #                                        prob_state=PROB_STATE_LABLES["IPD"], figsize=(10, 3),
    #                                        filename_to_save=folder +
    #                                                         "walk_de/lola_e" + str(e_idx) +
    #                                                         "d" + str(d_idx) + "_walk.pdf", top=[None, None])
    #     # ordered_results[g][p], intervals, STATES[game], prob_state = PROB_STATE_LABLES[game], figsize = (10, 3),
        # filename_to_save = folder + "my_" + pair + "_" + game + "_space_walk_all.pdf", top = [None, None])
    # titles[d_idx] = r"$\detla={0}$".format(delta)
    # init_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #
    # ordered_results = [None for _ in range(len(81))]
    # for filename, X in results.items():
    #     v = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
    #     print(v)
    #     val_idx = int(v)
    #     ordered_results[val_idx] = X
    #
    # for v, value in enumerate(init_values):


def lola_delta_eta_walk_through_space_figure(folder="../results/lola_robust_delta_eta/"):
    agent_pair = "lola1_vs_lola1"
    game = "IPD"

    ETA = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0, 15.0, 20.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10]

    de_pairs = np.array([[0.1, 20.0], [0.1, 10], [1.0, 2.0]])#[[0.1, 1.0], [1.0, 0.1], [5.0, 1.0], [1.0, 5.0]])#[[0.1, 5.0], [0.5, 0.5], [5.0, 5.0], [2.0, 0.1], [1.0, 1.0]])

    intervals = [[0, 4], [4, 9], [9, 49], [49, 199]]

    for pair in de_pairs:
        d = pair[0]
        e = pair[1]
        d_idx = int(DELTA.index(d))
        e_idx = int(ETA.index(e))
        subfolder = "D{0:02d}/E{1:02d}/".format(d_idx, e_idx) # "D0" + str(d_idx) + "/E0" + str(e_idx)
        results = collect_experiment_results(folder+subfolder, "*" + game + "*.json", ignore_root=False)
        # print(results.keys())
        # ordered_results = [[None] * len(ETA) for _ in range(len(DELTA))]
        for filename, X in results.items():
            delta = X["config"]["agent_pair"]["delta"]
            eta = X["config"]["agent_pair"]["eta"]
            delta_eta = filename.split(folder)[1].split("/")  # filename.split(folder)[1].split(".")[0][:-4] #
            if delta in de_pairs[:, 0] and eta in de_pairs[:, 1] and game in filename:
                e_idx = int(delta_eta[1][1:])
                d_idx = int(delta_eta[0][1:])
                plot_policy_walk_through_space(X, intervals, STATES["IPD"],
                                               prob_state=PROB_STATE_LABLES["IPD"], figsize=(9, 3),
                                               filename_to_save=folder+"walk_de/lola_e" + str(e_idx) +
                                                                "d" + str(d_idx) + "_walk.pdf", top=[None, 15])


def lola_randomness_robustness_metrics(folder="../results/lola_uniform_random_init_policy/"):
    agent_pair_order = ["lola1_vs_lola1"]
    game = "IPD"

    randomness = np.linspace(0, 0.5, 51)
    ordered_results = [None] * len(randomness)

    file = folder + "compressed_after_50_for_100.npy"

    results = collect_experiment_results(folder, "*{0}.json".format(game), top=1)
    my_X = list(results.values())[0]

    if not os.path.exists(file):
        results = collect_experiment_ith_policies(folder, 50, "*{0}.json".format(game))

        for filename, X in results.items():
            r = int(filename.split(folder)[1].split("/")[0][1:])
            ordered_results[r] = X

        ordered_results = np.array(ordered_results)
        np.save(file, ordered_results)
    else:
        ordered_results = np.load(file)

    metrics = [(["R", "R (agent 1)"], lambda x1, x2: R(x1, x2, gamma=my_X["config"]["agent_pair"]["gamma"],
                                                       r1=my_X["config"]["game"]["payoff1"],
                                                       r2=my_X["config"]["game"]["payoff2"]), 0, ["red", "orange"], True),
               (["%TFT"], lambda x1, x2: tft(x1, x2), 1, ["red"], True),
               (["%TFT2"], lambda x1, x2: tft2(x1, x2), 1, ["blue"], True),
               (["E[Y(CC)]"], lambda x1, x2: exp_s(x1, x2, state=1), 2, ["blue"], True),
               (["E[Y(CD)]"], lambda x1, x2: exp_s(x1, x2, state=2), 2, ["orange"], True),
               (["E[Y(DC)]"], lambda x1, x2: exp_s(x1, x2, state=3), 2, ["green"], True),
               (["E[Y(DD)]"], lambda x1, x2: exp_s(x1, x2, state=4), 2, ["red"], True)]
    ylabels = ["Av. reward per step, R", "Av. % of TFT policy", "Expected state visits, E[Y(s)]"]

    my_results = []
    for rr in range(ordered_results.shape[1]):
        my_results.append(ordered_results[:, rr])

    yticks = [np.linspace(-1.5, -1, 6), np.linspace(50, 100, 6), np.linspace(0, 100, 6)]
    ybounds = [(-1.5-0.05*0.5, -1+0.05*0.5), (50-0.05*50, 100+0.05*50), (0-5, 100+5)]

    plot_metrics_across_x(np.array(my_results), metrics, filename=folder + "randomness_r_on_R_TFT_EY_after_50_for_100.pdf",
                          n_metrics=3, show=False, ylabels=ylabels, xticks=randomness, yticks=yticks, ybounds=ybounds,
                          xlabels=["Randomness, r, in initial policy probability drawn from a uniform distribution of [0.5-r, 0.5+r]"]*3)


def generate_metric_table(results_array, metric, y_lbls, x_lbls):

    line = ""
    for _, x in enumerate(x_lbls):
        line += " & {0:.2f}".format(x)
    print(line, end=r"\\ \hline" + "\n")
    line = ""
    for i, y in reversed(list(enumerate(y_lbls))):
        for j, x in enumerate(x_lbls):
            if j == 0:
                line = "{0:.2f} & ".format(y)
            else:
                line += " & "

            S = y
            T = x
            if S < 0 and T > 1:
                # Prisoners Dilemma
                line += "\cellcolor[HTML]{FFDFDE}"
            elif S < 0 and T <= 1:
                # Stag hunt
                line += "\cellcolor[HTML]{CCEACC}"
            elif S >= 0 and T > 1:
                # Snowdrift
                if S + T < 2:
                    line += "\cellcolor[HTML]{A9D5FF}"
                else:
                    line += "\cellcolor[HTML]{CBE6FF}"
            else:
                # Harmony Game
                if S + T < 2:
                    line += "\cellcolor[HTML]{D2D2D2}"
                else:
                    line += "\cellcolor[HTML]{E7E7E7}"

            end_policies = results_array[i][j]
            (R1, conf_R1) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1], join_policies=True,
                                                             # conf_interval=0.95,
                                                             std=True,
                                                             metric_fn=metric["function"])

            if "function2" in metric:
                (conf_R1, _) = get_av_metrics_for_policy_arrays(end_policies[:, 0], end_policies[:, 1],
                                                                 join_policies=True,
                                                                 # conf_interval=0.95,
                                                                 std=True,
                                                                 metric_fn=metric["function2"])

                line += "{0:.2f}".format(R1) + " : {0:.2f}".format(conf_R1)
            else:
                line += r"\textbf{" + "{0:.2f}".format(R1) + "}"+"({0:.2f})".format(conf_R1)

        print(line, end=r"\\ \hline" + "\n")
        line = ""


def plot_lola_through_st_space_table(folder="../results/lola_through_ST_space/"):
    agent_pair = "lola1b_vs_lola1"  # "lola1b_vs_lola1", "lola1b_vs_lola1b"
    game = "unspecified"

    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    ordered_results = [[None] * len(T) for _ in S]
    file = folder + "compressed_{0}_new.npy".format(agent_pair)
    pattern = "*{0}_{1}.json".format(agent_pair, game)
    results = collect_experiment_results(folder, pattern, top=None)
    my_X = list(results.values())[0]

    if not os.path.exists(file):
        results = collect_experiment_end_policies(folder, pattern)

        for filename, X in results.items():
            s = int(filename.split(folder)[1].split("/")[0][1:])
            t = int(filename.split(folder)[1].split("/")[1][1:])
            ordered_results[s][t] = X

        ordered_results = np.array(ordered_results)
        np.save(file, ordered_results)
    else:
        ordered_results = np.load(file)

    metric = {"title": "R (std)",
              # "function": lambda x1, x2: R(x1, x2, gamma=my_X["config"]["agent_pair"]["gamma"],
              #                                          r1=my_X["config"]["game"]["payoff1"],
              #                                          r2=my_X["config"]["game"]["payoff2"])}
              "function": lambda x1, x2: exp_s(x1, x2, state=5)}
              # "function": lambda x1, x2: exp_s(x1, x2, state=1),
              # "function2": lambda x1, x2: exp_s(x1, x2, state=5)}

    generate_metric_table(ordered_results, metric, S, T)


# lola_through_ST_space/S06/T05/lola1_vs_lola1_unspecified.json
def basic_policies_and_R_through_time(folder="../results/lola_through_ST_space/S03/T00/"):

    results = collect_experiment_results(folder, "*.json", ignore_root=False)

    agent_pair_order = ["nl_vs_nl", "lola1_vs_lola1"]
    #                     , "lola1_vs_nl", "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    # game = "ISD"  # "ISH", "ISD",
    settings = dict()

    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    s = int(folder.split("/")[-3:][0][1:])
    t = int(folder.split("/")[-3:][1][1:])

    print(S[s], T[t])

    maybe = ""
    if S[s] < 0 and T[t] > 1:
        # Prisoners Dilemma
        game = "IPD"
    elif S[s] < 0 and T[t] <= 1:
        # Stag hunt
        game = "ISH"
    elif S[s] >= 0 and T[t] > 1:
        # Snowdrift
        game = "ISD"
    else:
        game = "IPD"
        maybe = "_harmony"

    ordered_results = [None for _ in range(len(agent_pair_order))]
    base_folder = os.path.join(*folder.split("/")[:2])
    folder_filename = base_folder + "/ey/my_" + game + maybe+"_" + "_".join(folder.split("/")[-3:]) + ".pdf"
    print(folder_filename)
    for filename, X in results.items():
        # game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        # pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        pair = X["config"]["simulation"]["agent_pair"]
        if  pair in agent_pair_order:
            # game_idx = game_order.index(game)
            pair_idx = agent_pair_order.index(pair)
            ordered_results[pair_idx] = X
            settings = dict(gamma=X["config"]["agent_pair"]["gamma"],
                                      r1=X["config"]["game"]["payoff1"],
                                      r2=X["config"]["game"]["payoff2"])

    # for g, game in enumerate(game_order):
    plot_v_timeline_st_space_id(ordered_results, STATES[game], filename_to_save=folder_filename,
                                 prob_state=PROB_STATE_LABLES[game], **settings, top=50)


# lola_through_ST_space/S06/T05/lola1_vs_lola1_unspecified.json
def basic_EY_through_time(folder="../results/lola_through_ST_space/S03/T00/"):
    agent_pair_order = ["lola1b_vs_lola1"]

    results = collect_experiment_results(folder, "*" + agent_pair_order[0] + "*.json", ignore_root=False)

    #                     , "lola1_vs_nl", "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    # game = "ISD"  # "ISH", "ISD",
    settings = dict()

    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    s = int(folder.split("/")[-3:][0][1:])
    t = int(folder.split("/")[-3:][1][1:])

    print(S[s], T[t])

    maybe = ""
    if S[s] < 0 and T[t] > 1:
        # Prisoners Dilemma
        game = "IPD"
    elif S[s] < 0 and T[t] <= 1:
        # Stag hunt
        game = "ISH"
    elif S[s] >= 0 and T[t] > 1:
        # Snowdrift
        game = "ISD"
    else:
        game = "IPD"
        maybe = "_harmony"

    ordered_results = [None for _ in range(len(agent_pair_order))]
    base_folder = os.path.join(*folder.split("/")[:3])
    folder_filename = base_folder + "/ey_metric/my_" + game + maybe+"_" + "_".join(folder.split("/")[-3:]) + ".pdf"
    print(folder_filename)
    for filename, X in results.items():
        # game = filename.split(folder)[1].split("/")[0]  # filename.split(folder)[1].split(".")[0][-3:] #
        # pair = filename.split(folder)[1].split("/")[1]  # filename.split(folder)[1].split(".")[0][:-4] #
        pair = X["config"]["simulation"]["agent_pair"]
        if  pair in agent_pair_order:
            # game_idx = game_order.index(game)
            pair_idx = agent_pair_order.index(pair)
            ordered_results[pair_idx] = X
            settings = dict(gamma=X["config"]["agent_pair"]["gamma"],
                                      r1=X["config"]["game"]["payoff1"],
                                      r2=X["config"]["game"]["payoff2"])

    # for g, game in enumerate(game_order):
    plot_ey_timeline_st_space_id(ordered_results, STATES[game], filename_to_save=folder_filename,
                                 prob_state=PROB_STATE_LABLES[game], **settings, top=50)


if __name__ == "__main__":
    # table_basic_experiments()
    # basic_experiment_replications_table()
    # basic_experiment_replications_figure()
    # basic_experiment_replications_walk_through_space_figure()
    # plot_sigmoid()
    # lola_robust_delta_eta()
    # lola_robust_delta_eta_policies_grid()
    # lola_robust_delta_eta_R_conv_tft2_graph()
    # metrics_through_time_graph()
    # lola_single_value_policy_init_walk_through_space_figure()
    # plot_st_space()
    # lola_randomness_robustness_metrics()
    # plot_lola_through_st_space_table()
    # lola_delta_eta_walk_through_space_figure()

    for i in range(9)[6:]:
        for j in range(9)[5:]:
            path = "../results/lola_through_ST_space/S0{0}/T0{1}/".format(i, j)
            print(path)
            basic_EY_through_time(folder=path)
