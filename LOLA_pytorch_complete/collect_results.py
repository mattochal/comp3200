from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json

from LOLA_pytorch import LOLA_vs_LOLA
from LOLA_pytorch import LOLAOM_vs_LOLAOM


def run_experiment(repeats=5):
    results=defaultdict(lambda: [])
    for ex in range(repeats):
        print("Experiment: ", ex)
        p1, p2, _ = LOLAOM_vs_LOLAOM.run(n=200)
        results["LOLAOM_vs_LOLAOM"].append([p1.data.numpy(), p2.data.numpy()])

    for experiment in results.keys():
        X = np.array(results[experiment])
        results[experiment] = np.squeeze(X).tolist()

    return results


def show_results(results):
    fig, ax = plt.subplots()
    for experiment, X in results.items():
        X = np.array(X)
        colors = ["purple", "blue", "green", "orange", "red"]
        state = ["s0", "CC", "CD", "DC", "DD"]
        for s in range(5):
            ax.scatter(X[:, 0, s], X[:, 1, s], s=55, c=colors[s], alpha=0.5, label=state[s])
        plt.title(experiment)
        ax.legend(loc='best', shadow=True)
        plt.show()


def save_results(results):
    with open('stag.json', 'w') as outfile:
        json.dump(results, outfile)


def load_results(filename='stag.json'):
    # Writing JSON data
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    results = run_experiment()
    save_results(results)
    # results = load_results()
    show_results(results)
