import subprocess
import itertools
import re

exec_file = "archvision/run.py"
seeds = 3

# queue, cores, use_gpu = "gpu_rtx", 5, True
queue, cores, use_gpu = "gpu_a100", 12, True
# queue, cores, use_gpu = "local", 4, False

if(queue == "local" and use_gpu == True):
    raise Exception ("No GPUs available on this partition!")

configs = {
    "exp_name": ["arch_params"],
    "model.nonlin": ["none", "relu", "elu", "tanh", "sigmoid"],
    "model.weights_init": ["kaiming", "kaiming_uniform", "xavier", "xavier_uniform", "gaussian", "uniform"],
    "model.norm": ["none", "batch", "channel", "instance"],
    "log_expdata": [True],
}

# configs = {
#     "exp_name": ["benchmark"],
#     "model.name": ["densenet121"],
#     "model.pretrained": [True, False],
#     "log_expdata": [True],
# }
# function to iterate through all values of dictionary:
combinations = list(itertools.product(*configs.values()))

# generate config string to pass to bash script
use_gpu = str(use_gpu).lower()
for combination in combinations:
    execstr = "python " + f"{exec_file}"
    for idx, key in enumerate(configs.keys()):
        execstr += " " + key + "=" + str(combination[idx])
    cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]

    # Run the command and capture the output
    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(output)
