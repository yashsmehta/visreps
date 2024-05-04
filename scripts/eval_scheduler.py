import subprocess
import itertools
import numpy as np
import re

exec_file = "archvision/run_eval.py"
seeds = 1 

queue, cores, use_gpu = "gpu_rtx8000", 6, True
# queue, cores, use_gpu = "gpu_rtx", 5, True
# queue, cores, use_gpu = "gpu_a100", 12, True
# queue, cores, use_gpu = "gpu_h100", 12, True
# queue, cores, use_gpu = "local", 4, False

if(queue == "local" and use_gpu == True):
    raise Exception ("No GPUs available on this partition!")

configs = {
    "cfg_id": list(range(13, 31)),
    "epoch": [0, 10, 20, 30, 40],
}


combinations = list(itertools.product(*configs.values()))

use_gpu = str(use_gpu).lower()
for combination in combinations:
    execstr = "python " + f"{exec_file}"
    for idx, key in enumerate(configs.keys()):
        execstr += " " + key + "=" + str(combination[idx])
    cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]

    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(output)
