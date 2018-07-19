import json
from subprocess import call
import os
import numpy as np
import math
from default_game_payoffs import default_payoffs


WORKING_DIR = ""


def humanize_time(secs, mins_roundup=10):
    mins, _ = divmod(secs, 60)
    mins = math.ceil(float(mins)/mins_roundup)*mins_roundup
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, 0)


def substitute(json_input, params):
    def subst(base_json, path, json_to_sub):
        if len(path) == 1:
            base_json[path[0]] = json_to_sub
            return base_json
        else:
            base_json[path[0]] = subst(base_json[path[0]], path[1::], json_to_sub)
            return base_json

    for p in params:
        json_input = subst(json_input.copy(), p.split("=")[0].strip().split("."), json.loads(p.split("=")[1]))

    return json_input


def load_json(config_path):
    with open(config_path) as data_file:
        json_input = json.load(data_file)
    return json_input


def save_config(path_to_file, json_output):
    with open(path_to_file, 'w') as outfile:
        print("Saving to " + path_to_file)
        json.dump(json_output, outfile, sort_keys=True, indent=4, separators=(',', ': '))


def invoke_qsub(program, flags, output_stream, walltime):
    flags_str = ""
    for f in flags:
        flags_str += f + " "

    instr = """
#PBS -l walltime={3}
PBS_O_WORKDIR="comp3200/"
cd $PBS_O_WORKDIR
echo "path"
echo $PBS_O_WORKDIR

module load numpy
module load python/3.5.1
module load conda/4.3.21
conda create --name mypytorch python=3.5 <<< $'y'
source activate mypytorch
pip install torch

unset PYTHONPATH

echo "Running {0} {1}"
echo "Going to save output log to {2}"
python {0} {1} > {2}""".format(program, flags_str, output_stream + "_log", walltime)

    qsub_instr_file = output_stream + "_run"
    with open(qsub_instr_file, 'w') as file:
        file.write(instr)

    instr = ["qsub", qsub_instr_file]
    print(instr)
    call(instr)


def invoke_bash(program, flags, output_stream):
    flags_str = ""
    for f in flags:
        flags_str += f + " "

    instr = """/Users/mateuszochal/.virtualenvs/3rdYearProject/bin/python {0} {1} """\
        .format(program, flags_str)

    qsub_instr_file = output_stream + "_run"
    with open(qsub_instr_file, 'w') as file:
        file.write(instr)

    instr = ["bash", qsub_instr_file]
    print(instr)
    call(instr)


def invoke_dilemmas_qsubs(output_stream, other_flags, params, agent_pair, walltime):
    dilemmas = ["IPD", "ISD", "ISH"]
    for d in dilemmas:
        invoke_dilemma_qsubs(d, output_stream, other_flags, params, agent_pair, walltime)


def invoke_dilemma_qsubs(d, output_stream, other_flags, params, agent_pair, walltime):
    flags = other_flags[:]
    flags.extend(["-p"])
    flags.extend(params)

    # DON'T USE (for reference only):
    # call(["/Users/mateuszochal/.virtualenvs/3rdYearProject/bin/python", "simulation.py", *flags])
    
    if TEST:
        invoke_bash("simulation.py", flags, output_stream + "" + agent_pair + "_" + d)
    else:
        invoke_qsub("simulation.py", flags, output_stream + "" + agent_pair + "_" + d, walltime)


def rollouts_small(folder="rollouts_small/"):
    rollouts(folder, num_rollouts=[5, 10, 15, 20], rollout_lengths=[5, 10, 15, 20], factor=0.0)


def rollouts(folder="rollouts/", num_rollouts=[25, 50, 75, 100], rollout_lengths = [20, 50, 100, 150], factor=0.029):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    if TEST:
        repeats = 2
        epochs = 2
    else:
        repeats = 50
        epochs = 200

    wall_time_offset = 15*60
    agent_pair = AGENT_PAIR

    for num in num_rollouts:
        for length in rollout_lengths:
            sub_folder = path_to_folder + "{0:03d}x{1:03d}/".format(num, length)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset + factor * (num*length - 25.0*20) * repeats)
            flags = ["-o", sub_folder, "-i", path_to_config]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'agent_pairs.{0}.rollout_length = {1}'""".format(agent_pair, length),
                      """'agent_pairs.{0}.num_rollout = {1}'""".format(agent_pair, num),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair))]
            invoke_dilemmas_qsubs(sub_folder, flags, params, epochs, agent_pair=agent_pair, walltime=wall_time)


def ST_space(folder="ST_space/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    R = 1.0
    P = 0.0
    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    if TEST:
        repeats = 2
        num = 2
        length = 2
        epochs = 2
    else:
        repeats = 50
        num = 50
        length = 50
        epochs = 200

    wall_time_offset = 60*60
    factor = 0
    agent_pair = AGENT_PAIR
    game = "IPD"

    for i, s in enumerate(S):
        for j, t in enumerate(T):
            sub_folder = path_to_folder + "S{0:02d}xT{1:02d}/".format(i, j)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset + factor * (num*length - 25.0*20) * repeats)
            flags = ["-o", sub_folder, "-i", path_to_config]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'agent_pairs.{0}.rollout_length = {1}'""".format(agent_pair, length),
                      """'agent_pairs.{0}.num_rollout = {1}'""".format(agent_pair, num),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      """'games.{0}.payoff1 = {1}'""".format(game, json.dumps([R, s, t, P])),
                      """'games.{0}.payoff2 = {1}'""".format(game, json.dumps([R, t, s, P]))]
            invoke_dilemma_qsubs(game, sub_folder, flags, params, epochs, agent_pair=agent_pair, walltime=wall_time)


def IPD_SG_space(folder="IPD_SG_space/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    R = 1.0
    P = 0.0
    S = np.linspace(-1.0, 0.0, num=9)
    T = 2.0

    Gammas = np.linspace(0.0, 1.0, num=11)
    Gammas[10] = 0.99  # if 1 then singular matrix - not good

    if TEST:
        repeats = 2
        num = 2
        length = 2
        epochs = 2
    else:
        repeats = 50
        num = 50
        length = 50
        epochs = 200

    wall_time_offset = 60 * 60
    factor = 0
    agent_pair = AGENT_PAIR
    game = "IPD"

    for i, s in enumerate(S):
        for j, g in enumerate(Gammas):
            sub_folder = path_to_folder + "S{0:02d}xG{1:02d}/".format(i, j)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset + factor * (num * length - 25.0 * 20) * repeats)
            flags = ["-o", sub_folder, "-i", path_to_config]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'agent_pairs.{0}.rollout_length = {1}'""".format(agent_pair, length),
                      """'agent_pairs.{0}.num_rollout = {1}'""".format(agent_pair, num),
                      """'agent_pairs.{0}.gamma = {1}'""".format(agent_pair, g),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      """'games.{0}.payoff1 = {1}'""".format(game, json.dumps([R, s, T, P])),
                      """'games.{0}.payoff2 = {1}'""".format(game, json.dumps([R, T, s, P]))]
            invoke_dilemma_qsubs(game, sub_folder, flags, params, epochs, agent_pair=agent_pair, walltime=wall_time)


def policy_init(folder="policy_init/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    theta1 = np.linspace(0.0, 1.0, num=9)
    theta2 = np.linspace(0.0, 1.0, num=9)

    if TEST:
        repeats = 2
        num = 2
        length = 2
        epochs = 2
    else:
        repeats = 50
        num = 50
        length = 50
        epochs = 200

    wall_time_offset = 60 * 60
    factor = 0
    agent_pair = AGENT_PAIR
    game = "IPD"

    for i, t1 in enumerate(theta1):
        for j, t2 in enumerate(theta2):
            sub_folder = path_to_folder + "P{0:02d}xP{1:02d}/".format(i, j)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset + factor * (num*length - 25.0*20) * repeats)
            flags = ["-o", sub_folder, "-i", path_to_config]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'agent_pairs.{0}.rollout_length = {1}'""".format(agent_pair, length),
                      """'agent_pairs.{0}.num_rollout = {1}'""".format(agent_pair, num),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      """'games.{0}.init_policy1 = {1}'""".format(game, json.dumps([t1]*5)),
                      """'games.{0}.init_policy2 = {1}'""".format(game, json.dumps([t2]*5))]
            invoke_dilemma_qsubs(game, sub_folder, flags, params, epochs, agent_pair=agent_pair, walltime=wall_time)


def long_epochs(folder="long_epochs/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    ETA = [0.01, 0.1, 0.5, 1.0, 5, 7.5, 10, 15]
    DELTA = [0.0005, 0.001, 0.01, 0.1, 0.25, 0.5, 1.0, 3.0]

    if TEST:
        repeats = 5
        num = 5
        length = 5
        epochs = 2
    else:
        repeats = 50
        num = 50
        length = 50
        epochs = 1000

    wall_time_offset = 4 * 60 * 60
    factor = 0
    agent_pair = AGENT_PAIR
    game = "IPD"

    for i, eta in enumerate(ETA):
        for j, delta in enumerate(DELTA):
            sub_folder = path_to_folder + "E{0:02d}xD{1:02d}/".format(i, j)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset + factor * (num * length - 25.0 * 20) * repeats)
            flags = ["-o", sub_folder, "-i", path_to_config]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'agent_pairs.{0}.rollout_length = {1}'""".format(agent_pair, length),
                      """'agent_pairs.{0}.num_rollout = {1}'""".format(agent_pair, num),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      """'agent_pairs.{0}.eta = {1}'""".format(agent_pair, eta),
                      """'agent_pairs.{0}.delta = {1}'""".format(agent_pair, delta),
                      """'games.{0}.init_policy1 = {1}'""".format(game, json.dumps([0.5] * 5)),
                      """'games.{0}.init_policy2 = {1}'""".format(game, json.dumps([0.5] * 5))]
            invoke_dilemma_qsubs(game, sub_folder, flags, params, epochs, agent_pair=agent_pair, walltime=wall_time)


def randomness_robustness(folder="random_init_policy_robustness/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    if TEST:
        repeats = 5
        num = 5
        length = 5
        epochs = 2
    else:
        repeats = 1
        num = 50
        length = 50
        epochs = 1000

    if AGENT_PAIR == "lola1_vs_lola1":
        wall_time_offset = 1.5 * 60 * 60

    elif AGENT_PAIR == "lola1b_vs_lola1b":
        wall_time_offset = 2 * 60 * 60

    factor = 0
    agent_pair = AGENT_PAIR
    game = "IPD"

    randomness = np.linspace(0, 0.5, 51)

    eta = 5
    delta = 0.25
    gamma = 0.96

    for i, r in enumerate(randomness):
        sub_folder = path_to_folder + "R{0:02d}/".format(i)
        os.makedirs(sub_folder, exist_ok=True)
        wall_time = humanize_time(wall_time_offset + factor * (num * length - 25.0 * 20) * repeats)
        flags = ["-o", sub_folder, "-i", path_to_config]

        dist = "{" + """"name": "uniform", "params": [{0}, {1}]""".format(0.5 - r, 0.5 + r) + "}"
        params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                  """'agent_pairs.{0}.rollout_length = {1}'""".format(agent_pair, length),
                  """'agent_pairs.{0}.num_rollout = {1}'""".format(agent_pair, num),
                  """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                  """'agent_pairs.{0}.eta = {1}'""".format(agent_pair, eta),
                  """'agent_pairs.{0}.delta = {1}'""".format(agent_pair, delta),
                  """'agent_pairs.{0}.gamma = {1}'""".format(agent_pair, gamma),
                  """'games.{0}.init_policy1 = {1}'""".format(game, json.dumps([None] * 5)),
                  """'games.{0}.init_policy2 = {1}'""".format(game, json.dumps([None] * 5)),
                  """'simulation.random_init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
        invoke_dilemma_qsubs(game, sub_folder, flags, params, epochs, agent_pair=agent_pair, walltime=wall_time)


def basic_experiments(folder="basic_experiments/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    if TEST:
        repeats = 1
        epoch_length = 5
        num_rollout = 0
        rollout_length = 0
    else:
        repeats = 1000
        epoch_length = 200
        num_rollout = 0
        rollout_length = 0

    agent_pairs = ["lola1_vs_lola1", "nl_vs_nl", "lola1_vs_nl", "lola1b_vs_nl", "lola1b_vs_lola1", "lola1b_vs_lola1b"]

    games = ["IPD", "ISH", "ISD", "IMP"]

    wall_time_offset = 2 * 60 * 60

    eta = 1
    delta = 1
    beta = 1
    gamma = 0.96

    for i, agent_pair in enumerate(agent_pairs):
        for j, game in enumerate(games):
            sub_folder = path_to_folder + "{0}/{1}/".format(game, agent_pair)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset)
            flags = ["-o", sub_folder, "-i", path_to_config]

            dist = "{" + """"name": "uniform", "params": [{0}, {1}]""".format(0, 1) + "}"
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'simulation.length = {0}'""".format(json.dumps(epoch_length)),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      """'simulation.game = {0}'""".format(json.dumps(game)),
                      """'game.payoff1 = {0}'""".format(json.dumps(default_payoffs[game]["payoff1"])),
                      """'game.payoff2 = {0}'""".format(json.dumps(default_payoffs[game]["payoff2"])),
                      """'agent_pair.rollout_length = {0}'""".format(rollout_length),
                      """'agent_pair.num_rollout = {0}'""".format(num_rollout),
                      """'agent_pair.eta = {0}'""".format(eta),
                      """'agent_pair.delta = {0}'""".format(delta),
                      """'agent_pair.beta = {0}'""".format(beta),
                      """'agent_pair.gamma = {0}'""".format(gamma),
                      """'agent_pair.init_policy1 = {0}'""".format(json.dumps([None] * 5)),
                      """'agent_pair.init_policy2 = {0}'""".format(json.dumps([None] * 5)),
                      """'agent_pair.init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
            invoke_dilemma_qsubs(game, sub_folder, flags, params, agent_pair=agent_pair, walltime=wall_time)


def basic_lola_replication(folder="basic_lola_replication_200_epochs/"):
    global TEST
    TEST = True
    FOLDER_PREFIX = "results/"

    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    repeats = 30
    epoch_length = 200
    num_rollout = 0
    rollout_length = 0

    agent_pairs = ["lola1_vs_lola1", "nl_vs_nl"]

    games = ["IPD", "IMP"]

    wall_time_offset = 0

    etas = {"IPD": 1, "IMP": 1}
    deltas = {"IPD": 1, "IMP": 1}

    sigmas = {"IPD": 1, "IMP": 1}
    gammas = {"IPD": 0.96, "IMP": 0.9}

    beta = 1

    for i, agent_pair in enumerate(agent_pairs):
        for j, game in enumerate(games):
            sub_folder = path_to_folder + "{0}/{1}/".format(game, agent_pair)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset)
            flags = ["-o", sub_folder, "-i", path_to_config]

            dist = "{" + """"name": "normal", "params": [{0}, {1}]""".format(0, sigmas[game]) + "}"
            eta = etas[game]
            delta = deltas[game]
            gamma = gammas[game]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                      """'simulation.length = {0}'""".format(json.dumps(epoch_length)),
                      """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      """'simulation.game = {0}'""".format(json.dumps(game)),
                      """'game.payoff1 = {0}'""".format(json.dumps(default_payoffs[game]["payoff1"])),
                      """'game.payoff2 = {0}'""".format(json.dumps(default_payoffs[game]["payoff2"])),
                      """'agent_pair.rollout_length = {0}'""".format(rollout_length),
                      """'agent_pair.num_rollout = {0}'""".format(num_rollout),
                      """'agent_pair.eta = {0}'""".format(eta),
                      """'agent_pair.delta = {0}'""".format(delta),
                      """'agent_pair.beta = {0}'""".format(beta),
                      """'agent_pair.gamma = {0}'""".format(gamma),
                      """'agent_pair.init_policy1 = {0}'""".format(json.dumps([None] * 5)),
                      """'agent_pair.init_policy2 = {0}'""".format(json.dumps([None] * 5)),
                      """'agent_pair.init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
            invoke_dilemma_qsubs(game, sub_folder, flags, params, agent_pair=agent_pair, walltime=wall_time)


def lola_single_value_policy_init(folder="lola_single_value_policy_init/"):
    global TEST
    TEST = True
    FOLDER_PREFIX = "results/"

    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    repeats = 1
    epoch_length = 500
    num_rollout = 0
    rollout_length = 0

    agent_pairs = ["lola1_vs_lola1"]

    games = ["IPD"]

    wall_time_offset = 0

    etas = {"IPD": 100, "IMP": 1}
    deltas = {"IPD": 0.01, "IMP": 1}

    sigmas = {"IPD": 1, "IMP": 1}
    gammas = {"IPD": 0.96, "IMP": 0.9}

    beta = 1

    agent_pair = agent_pairs[0]
    game = games[0]

    init_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for v, value in enumerate(init_values):
        sub_folder = path_to_folder + "{0}/".format(v)
        os.makedirs(sub_folder, exist_ok=True)
        wall_time = humanize_time(wall_time_offset)
        flags = ["-o", sub_folder, "-i", path_to_config]

        dist = "{" + """"name": "normal", "params": [{0}, {1}]""".format(0, sigmas[game]) + "}"
        eta = etas[game]
        delta = deltas[game]
        gamma = gammas[game]
        params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                  """'simulation.length = {0}'""".format(json.dumps(epoch_length)),
                  """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                  """'simulation.game = {0}'""".format(json.dumps(game)),
                  """'game.payoff1 = {0}'""".format(json.dumps(default_payoffs[game]["payoff1"])),
                  """'game.payoff2 = {0}'""".format(json.dumps(default_payoffs[game]["payoff2"])),
                  """'agent_pair.rollout_length = {0}'""".format(rollout_length),
                  """'agent_pair.num_rollout = {0}'""".format(num_rollout),
                  """'agent_pair.eta = {0}'""".format(eta),
                  """'agent_pair.delta = {0}'""".format(delta),
                  """'agent_pair.beta = {0}'""".format(beta),
                  """'agent_pair.gamma = {0}'""".format(gamma),
                  """'agent_pair.init_policy1 = {0}'""".format(json.dumps([value] * 5)),
                  """'agent_pair.init_policy2 = {0}'""".format(json.dumps([value] * 5)),
                  """'agent_pair.init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
        invoke_dilemma_qsubs(game, sub_folder, flags, params, agent_pair=agent_pair, walltime=wall_time)


def lola_robust_delta_eta(folder="lola_robust_delta_eta/"):
    global TEST
    TEST = True
    FOLDER_PREFIX = "results/"

    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    repeats = 50
    epoch_length = 300
    num_rollout = 0
    rollout_length = 0
    beta = 0

    agent_pairs = ["lola1_vs_lola1"]

    games = ["IPD"]

    sigmas = {"IPD": 1, "IMP": 1}
    gammas = {"IPD": 0.96, "IMP": 0.9}

    wall_time_offset = 60 * 60 * 1

    ETA = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0, 15.0, 20.0]
    DELTA = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10]
    # ETA = np.linspace(0.1, 2.0, 20)
    # DELTA = np.linspace(0.1, 2.0, 20)

    for i, agent_pair in enumerate(agent_pairs):
        for j, game in enumerate(games):
            for e, eta in enumerate(ETA):
                for d, delta in enumerate(DELTA):
                    if not( eta in [10.0, 15.0, 20.0] and delta in [0.1, 1.0]):
                        break
                    sub_folder = path_to_folder + "D{0:02d}/E{1:02d}/".format(d, e)
                    os.makedirs(sub_folder, exist_ok=True)
                    wall_time = humanize_time(wall_time_offset)
                    flags = ["-o", sub_folder, "-i", path_to_config]
                    dist = "{" + """"name": "normal", "params": [{0}, {1}]""".format(0, sigmas[game]) + "}"
                    gamma = gammas[game]
                    params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                              """'simulation.length = {0}'""".format(json.dumps(epoch_length)),
                              """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                              """'simulation.game = {0}'""".format(json.dumps(game)),
                              """'game.payoff1 = {0}'""".format(json.dumps(default_payoffs[game]["payoff1"])),
                              """'game.payoff2 = {0}'""".format(json.dumps(default_payoffs[game]["payoff2"])),
                              """'agent_pair.rollout_length = {0}'""".format(rollout_length),
                              """'agent_pair.num_rollout = {0}'""".format(num_rollout),
                              """'agent_pair.eta = {0}'""".format(eta),
                              """'agent_pair.delta = {0}'""".format(delta),
                              """'agent_pair.beta = {0}'""".format(beta),
                              """'agent_pair.gamma = {0}'""".format(gamma),
                              """'agent_pair.init_policy1 = {0}'""".format(json.dumps([None] * 5)),
                              """'agent_pair.init_policy2 = {0}'""".format(json.dumps([None] * 5)),
                              """'agent_pair.init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
                    invoke_dilemma_qsubs(game, sub_folder, flags, params, agent_pair=agent_pair, walltime=wall_time)


def lola_through_ST_space(folder="lola_through_ST_space/"):
    global TEST
    TEST = True
    FOLDER_PREFIX = "results/"

    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    if not TEST:
        repeats = 3
        epoch_length = 2
    else:
        repeats = 50
        epoch_length = 100

    num_rollout = 0
    rollout_length = 0
    beta = 0

    agent_pairs = ["nl_vs_nl"]  # , "lola1b_vs_lola1", "lola1b_vs_lola1b"]
    sigma = 1  # for normal dist
    gamma = 0.96
    eta = 1
    delta = 1
    R = 1.0
    P = 0.0
    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    wall_time_offset = 60 * 60 * 0.5
    game = "unspecified"

    for i, agent_pair in enumerate(agent_pairs):
        for j, s in enumerate(S):
            for k, t in enumerate(T):
                sub_folder = path_to_folder + "S{0:02d}/T{1:02d}/".format(j, k)
                os.makedirs(sub_folder, exist_ok=True)
                wall_time = humanize_time(wall_time_offset)
                flags = ["-o", sub_folder, "-i", path_to_config]
                dist = "{" + """"name": "normal", "params": [{0}, {1}]""".format(0, sigma) + "}"
                params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                          """'simulation.length = {0}'""".format(json.dumps(epoch_length)),
                          """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                          """'simulation.game = {0}'""".format(json.dumps(game)),
                          """'game.payoff1 = {0}'""".format([R, s, t, P]),
                          """'game.payoff2 = {0}'""".format([R, t, s, P]),
                          """'agent_pair.rollout_length = {0}'""".format(rollout_length),
                          """'agent_pair.num_rollout = {0}'""".format(num_rollout),
                          """'agent_pair.eta = {0}'""".format(eta),
                          """'agent_pair.delta = {0}'""".format(delta),
                          """'agent_pair.beta = {0}'""".format(beta),
                          """'agent_pair.gamma = {0}'""".format(gamma),
                          """'agent_pair.init_policy1 = {0}'""".format(json.dumps([None] * 5)),
                          """'agent_pair.init_policy2 = {0}'""".format(json.dumps([None] * 5)),
                          """'agent_pair.init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
                invoke_dilemma_qsubs(game, sub_folder, flags, params, agent_pair=agent_pair, walltime=wall_time)


def lola_randomness_robustness(folder="lola_uniform_random_init_policy/"):
    path_to_folder = WORKING_DIR + FOLDER_PREFIX + folder
    path_to_config = WORKING_DIR + "config.json"

    agent_pairs = ["lola1_vs_lola1"]
    games = ["IPD"]
    wall_time_offset = 60 * 60 * 1

    etas = {"IPD": 1, "IMP": 1}
    deltas = {"IPD": 1, "IMP": 1}
    sigmas = {"IPD": 1, "IMP": 1}
    gammas = {"IPD": 0.96, "IMP": 0.9}

    agent_pair = agent_pairs[0]
    game = games[0]

    beta = 1
    eta = etas[game]
    delta = deltas[game]
    sigma = sigmas[game]
    gamma = gammas[game]

    if TEST:
        repeats = 3
        epoch_length = 4
        num_rollout = 0
        rollout_length = 0
    else:
        repeats = 1000
        epoch_length = 500
        num_rollout = 0
        rollout_length = 0

    randomness = np.linspace(0, 0.5, 51)

    for i, r in enumerate(randomness):
        sub_folder = path_to_folder + "R{0:02d}/".format(i)
        os.makedirs(sub_folder, exist_ok=True)
        wall_time = humanize_time(wall_time_offset)
        flags = ["-o", sub_folder, "-i", path_to_config]

        dist = "{" + """"name": "uniform", "params": [{0}, {1}]""".format(0.5 - r, 0.5 + r) + "}"
        params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats)),
                  """'simulation.length = {0}'""".format(json.dumps(epoch_length)),
                  """'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                  """'simulation.game = {0}'""".format(json.dumps(game)),
                  """'game.payoff1 = {0}'""".format(json.dumps(default_payoffs[game]["payoff1"])),
                  """'game.payoff2 = {0}'""".format(json.dumps(default_payoffs[game]["payoff2"])),
                  """'agent_pair.rollout_length = {0}'""".format(rollout_length),
                  """'agent_pair.num_rollout = {0}'""".format(num_rollout),
                  """'agent_pair.eta = {0}'""".format(eta),
                  """'agent_pair.delta = {0}'""".format(delta),
                  """'agent_pair.beta = {0}'""".format(beta),
                  """'agent_pair.gamma = {0}'""".format(gamma),
                  """'agent_pair.init_policy1 = {0}'""".format(json.dumps([None] * 5)),
                  """'agent_pair.init_policy2 = {0}'""".format(json.dumps([None] * 5)),
                  """'agent_pair.init_policy_dist = {0}'""".format(json.dumps(json.loads(dist)))]
        invoke_dilemma_qsubs(game, sub_folder, flags, params, agent_pair=agent_pair, walltime=wall_time)


# TEST = True
TEST = False

# AGENT_PAIR = "lolaom_vs_lolaom"
# FOLDER_PREFIX = "results/lolaom_"

# AGENT_PAIR = "lola_vs_lola"
# FOLDER_PREFIX = "results/lola_"

# AGENT_PAIR = "lola1_vs_lola1"
AGENT_PAIR = ""
FOLDER_PREFIX = "results/"
#
# AGENT_PAIR = "lola1b_vs_lola1b"
# FOLDER_PREFIX = "results/lola1b_"

if TEST:
    FOLDER_PREFIX = "test_" + FOLDER_PREFIX

if __name__ == "__main__":
    # ST_space()
    # rollouts()
    # IPD_SG_space()
    # policy_init()
    # rollouts_small()
    # random_init_long_epochs()
    # long_epochs()
    # randomness_robustness()
    # basic_experiments()
    # basic_lola_replication()
    lola_robust_delta_eta()
    # lola_through_ST_space()
    # lola_single_value_policy_init()
    # lola_randomness_robustness()
    pass
