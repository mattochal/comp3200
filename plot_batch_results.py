import matplotlib.pyplot as plt


def plot_policies(results, keys):
    keys
    fig, ax = plt.subplots()
    X = get_policies(results)
    colors = ["purple", "blue", "orange", "green", "red"]
    state = ["s0", "CC", "CD", "DC", "DD"]
    for s in range(5):
        ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=state[s])
    plt.title(results["config"]["simulation"]["agent_pair"] + " in " + results["config"]["simulation"]["game"])
    plt.xlabel('P(cooperation | state) for agent 0')
    plt.ylabel('P(cooperation | state) for agent 1')
    ax.legend(loc='best', shadow=True)
    plt.show()
    pass