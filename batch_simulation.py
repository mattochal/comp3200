import json
from subprocess import call
import os

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
conda create --name mypytorch_test2 python=3.5 <<< $'y'
source activate mypytorch_test2
conda install pytorch torchvision -c pytorch <<< $'y'

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

    instr = """/Users/mateuszochal/.virtualenvs/3rdYearProject/bin/python {0} {1} > {2}"""\
        .format(program, flags_str, output_stream + "_log")

    qsub_instr_file = output_stream + "_run"
    with open(qsub_instr_file, 'w') as file:
        file.write(instr)

    instr = ["bash", qsub_instr_file]
    print(instr)
    call(instr)


def invoke_dilemmas_qsubs(output_stream, other_flags, params, epochs, walltime):
    dilemmas = ["IPD", "ISD", "ISH"]
    agent_pair = "lolaom_vs_lolaom"
    for d in dilemmas:
        flags = other_flags[:]
        flags.extend(["-p", r"""'simulation.game = {0}'""".format(json.dumps(d)),
                      r"""'simulation.agent_pair = {0}'""".format(json.dumps(agent_pair)),
                      r"""'games.{0}.n = {1}'""".format(d, epochs)])
        flags.extend(params)
        # print(["/Users/mateuszochal/.virtualenvs/3rdYearProject/bin/python", "simulation.py", *flags])
        # call(["/Users/mateuszochal/.virtualenvs/3rdYearProject/bin/python", "simulation.py", *flags])
        invoke_bash("simulation.py", flags, output_stream + "" + agent_pair + "_" + d)
        # invoke_qsub("simulation.py", flags, output_stream + "" + agent_pair + "_" + d, walltime)


# experiment2 focuses on varying the high-to-low value job ratio between ranges of 0 and 0.2 probability
def lolaom_dilemmas(folder="lolaom_dilemmas/"):
    path_to_folder = WORKING_DIR + folder
    path_to_config = WORKING_DIR + "config.json"

    num_rollouts = [25, 50, 75, 100]
    rollout_lengths = [20, 50, 100, 150]
    # num_rollouts = [5, 5]
    # rollout_lengths = [10, 20]
    repeats = 25
    epochs = 200

    wall_time_offset = 15*60
    factor = 0.014

    for num in num_rollouts:
        for length in rollout_lengths:
            sub_folder = path_to_folder + "{0:03d}x{1:03d}/".format(num, length)
            os.makedirs(sub_folder, exist_ok=True)
            wall_time = humanize_time(wall_time_offset + factor * (num*length - 25.0*20) * repeats)
            flags = ["-o", sub_folder, "-i", path_to_config]
            params = ["""'simulation.repeats = {0}'""".format(json.dumps(repeats))]
            invoke_dilemmas_qsubs(sub_folder, flags, params, epochs, walltime=wall_time)


if __name__ == "__main__":
    lolaom_dilemmas()

