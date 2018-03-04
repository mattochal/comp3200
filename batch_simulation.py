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

    invoke_bash("simulation.py", flags, output_stream + "" + agent_pair + "_" + d)
    # invoke_qsub("simulation.py", flags, output_stream + "" + agent_pair + "_" + d, walltime)


def lolaom_dilemmas(folder="lolaom_dilemmas/"):
    path_to_folder = WORKING_DIR + folder
    path_to_config = WORKING_DIR + "config.json"

    num_rollouts = [25, 50, 75, 100]
    rollout_lengths = [20, 50, 100, 150]
    num_rollouts = [5, 5]
    rollout_lengths = [10, 20]
    repeats = 5
    epochs = 2

    wall_time_offset = 15*60
    factor = 0.0145*2
    agent_pair = "lolaom_vs_lolaom"

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


def lolaom_ST_space(folder="lolaom_ST_space/"):
    path_to_folder = WORKING_DIR + folder
    path_to_config = WORKING_DIR + "config.json"

    R = 1.0
    P = 0.0
    S = np.linspace(-1.0, 1.0, num=9)
    T = np.linspace(0.0, 2.0, num=9)

    repeats = 50
    num = 50
    length = 50
    epochs = 200

    # repeats = 5
    # num = 5
    # length = 5
    # epochs = 2

    wall_time_offset = 60*60
    factor = 0
    agent_pair = "lolaom_vs_lolaom"
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


if __name__ == "__main__":
    lolaom_ST_space()
    # lolaom_dilemmas()
    pass
