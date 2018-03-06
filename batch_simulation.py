import json
from subprocess import call
import os
import numpy as np
import math

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


def invoke_dilemmas_qsubs(output_stream, other_flags, params, epochs, agent_pair, walltime):
    dilemmas = ["IPD", "ISD", "ISH"]
    for d in dilemmas:
        invoke_dilemma_qsubs(d, output_stream, other_flags, params, epochs, agent_pair, walltime)


def invoke_dilemma_qsubs(d, output_stream, other_flags, params, epochs, agent_pair, walltime):
    flags = other_flags[:]
    flags.extend(["-p",
                  """'simulation.game = {0}'""".format(json.dumps(d)),
                  """'games.{0}.n = {1}'""".format(d, epochs)])
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


def random_init_long_epochs(folder="random_init_long_epochs/"):
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

    wall_time_offset = 1 * 60 * 60
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
                      """'games.{0}.init_policy1 = {1}'""".format(game, json.dumps([None] * 5)),
                      """'games.{0}.init_policy2 = {1}'""".format(game, json.dumps([None] * 5))]
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

# TEST = True
TEST = False

# AGENT_PAIR = "lolaom_vs_lolaom"
# FOLDER_PREFIX = "results/lolaom_"

AGENT_PAIR = "lola_vs_lola"
FOLDER_PREFIX = "results/lola_"

if TEST:
    FOLDER_PREFIX = "test_" + FOLDER_PREFIX

if __name__ == "__main__":
    # ST_space()
    # rollouts()
    # IPD_SG_space()
    # policy_init()
    # rollouts_small()
    random_init_long_epochs()
    # long_epochs()
    pass
