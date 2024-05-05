import subprocess
import itertools


def main():
    """
    Execute training scripts on different configurations and hardware setups.

    This script sets up configurations for different hardware queues and executes
    a Python script for each configuration combination using a shell script to submit jobs.
    It handles different GPU settings and captures the output of each job submission.
    """
    exec_file = "archvision/run_train.py"
    seeds = 3

    queue, cores, use_gpu = "gpu_rtx8000", 6, True
    # queue, cores, use_gpu = "gpu_rtx", 5, True
    # queue, cores, use_gpu = "gpu_a100", 12, True
    # queue, cores, use_gpu = "local", 4, False

    if queue == "local" and use_gpu:
        raise Exception("No GPUs available on this partition!")

    configs = {
        "conv_trainable": ["11111"],
        "fc_trainable": ["111"],
        "use_wandb": [True],
        "data_augment": [False, True],
        "group": ["data_augmentation"],
        "exp_name": ["across_metrics"],
    }

    combinations = list(itertools.product(*configs.values()))
    use_gpu = str(use_gpu).lower()

    for combination in combinations:
        execstr = f"python {exec_file}"
        for idx, key in enumerate(configs):
            execstr += f" {key}={combination[idx]}"
        cmd = ["scripts/submit_job.sh", str(cores), str(seeds), queue, execstr, use_gpu]

        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, universal_newlines=True
        )
        print(output)


if __name__ == "__