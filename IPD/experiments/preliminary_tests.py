import numpy as np

from IPD.experiments import pgapp_vs_tft
from IPD.experiments import q_learner_vs_q_learner
from IPD.experiments import q_learner_vs_tft_ipd
from IPD.experiments import pgapp_vs_pgapp
from IPD.experiments import *
from collections import defaultdict
import json

results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def identify_end_strategy(a, last_n_moves=50, thresh=0.2):
    count = 0
    for transition in a.transitions[-last_n_moves:]:
        count += transition[1]

    count /= 1.0 * last_n_moves

    if count < thresh:
        return "C"
    if count > 1 - thresh:
        return "D"

    return "A"


def add_result(g, title, a1, a2):
    global results
    a1_strategy = identify_end_strategy(a1)
    a2_strategy = identify_end_strategy(a2)
    strategy = ''.join(sorted([a1_strategy, a2_strategy]))
    results[g][title][strategy] += 1


episodes = 100  # number of IPD games
n = 10000  # number of PD games per episode

for g in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9995]:
    print("\n\n----------------------- gamma = {0} -----------------------\n".format(g))
    for e in range(episodes):
        print("episode ", e)
        a1, a2, = q_learner_vs_tft_ipd.basic_game(n, g)
        add_result(g, "Q-agent vs TFT", a1, a2)

        # a1, a2, = pgapp_vs_tft.basic_game(n, g)
        # add_result(g, "PGA-PP vs TFT", a1, a2)
        #
        # a1, a2, = q_learner_vs_q_learner.basic_game(n, g)
        # add_result(g, "Q-agent vs Q-agent", a1, a2)
        #
        # a1, a2, = pgapp_vs_pgapp.basic_game(n, g)
        # add_result(g, "PGA-PP vs PGA-PP", a1, a2)

data = json.loads(json.dumps(results))

with open('results.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)