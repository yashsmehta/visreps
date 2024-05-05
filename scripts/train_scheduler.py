import subprocess
import itertools
import numpy as np
import re

exec_file = "archvision/run_train.py"
seeds = 3

queue, cores, use_gpu = "gpu_rtx8000", 6, True
# queue, cores, use_gpu = "gpu_rtx", 5, True
# queue, cores, use_gpu = "gpu_a100", 12, True
# queue, cores, use_gpu = "local", 4, False

if(queue == "local" and use_gpu == True):
    raise Exception ("No GPUs available on this partition!")

# configs = {
#     "conv_trainable": ["10000", "01000", "00100", "00010", "00001"],
#     "fc_trainable": ["000"],
#     "use_wandb": [True]
# }

configs = {
    "conv_trainable": ["11111"],
    "fc_trainable": ["111"],
    "use_wandb": [True],
    "data_augment": [False, True],
    "group": ["data_augmentation"],
    "exp_name": ["across_metrics"]
}

# function to iterate through all values of dictionary:
combinations = list(itertools.product(*configs.values()))

use_gpu = str(use_gpu).lower()
for combination in combinations:
    execstr = "python " + f"{exec_file}"
    for idx, key in enumerate(configs.keys()):
        execstr += " " + key + "=" + str(combination[idx])
    cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]
    print(cmd)
    exit()

    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(output)
