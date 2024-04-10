import subprocess
import itertools
import re

exec_file = "run.py"
seeds = 1 

# queue, cores, use_gpu = "gpu_rtx", 5, True
queue, cores, use_gpu = "gpu_tesla", 12, True
# queue, cores, use_gpu = "local", 4, False

if(queue == "local" and use_gpu == True):
    raise Exception ("No GPUs available on this partition!")

configs = {
    "exp_name": ["arch"],
    "nonlin": ["linear", "relu", "tanh", "sigmoid"],
    "init": ["kaiming", "xavier", "random"],
    "log_expdata": [True],
}
# function to iterate through all values of dictionary:
combinations = list(itertools.product(*configs.values()))

# generate config string to pass to bash script
use_gpu = str(use_gpu).lower()
for combination in combinations:
    execstr = "python " + f"{exec_file}"
    for idx, key in enumerate(configs.keys()):
        execstr += " " + key + "=" + str(combination[idx])
    execstr = re.sub(r'layer_sizes=(\[\d+,\s*\d+\])',
                        lambda m: f'"layer_sizes={m.group(1)}"',
                        execstr)
    cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]

    # Run the command and capture the output
    output = subprocess.check_output(
        cmd, stderr=subprocess.STDOUT, universal_newlines=True
    )
    print(output)
