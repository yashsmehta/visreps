#!/usr/bin/env python3
import os, subprocess

# --- Config ---
folders = ['alexnet_pca']
remote_base = "/scratch4/mbonner5/ymehta3/visreps/model_checkpoints"
local_base = "/data/ymehta3"
files_to_copy = ["checkpoint_epoch_20.pth", "config.json", "training_metrics.csv"]

# --- SSH setup ---
host = "rockfish"
control_path = "/tmp/ssh_mux_%h"

# Try keyless ssh to "rockfish" first; fall back to asking for username
test = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "true"],
                      capture_output=True)
if test.returncode == 0:
    ssh_target = host
else:
    user = input("Keyless SSH to 'rockfish' failed. Enter your Rockfish username: ").strip()
    ssh_target = f"{user}@{host}"

# Open persistent SSH connection
subprocess.run([
    "ssh", "-MNf",
    "-o", "ControlMaster=yes",
    "-o", f"ControlPath={control_path}",
    "-o", "ControlPersist=10m",
    ssh_target
])

# Copy files
for folder in folders:
    remote_folder = f"{remote_base}/{folder}"
    subdirs = subprocess.check_output(
        ["ssh", "-o", f"ControlPath={control_path}", ssh_target,
         f"ls -d {remote_folder}/cfg2a {remote_folder}/cfg4a 2>/dev/null"],
        text=True
    ).split()

    for subdir in subdirs:
        cfg = os.path.basename(subdir)
        local_dir = f"{local_base}/{folder}/{cfg}"
        os.makedirs(local_dir, exist_ok=True)
        for f in files_to_copy:
            src = f"{ssh_target}:{subdir}/{f}"
            print(f"Copying {src} → {local_dir}")
            subprocess.run([
                "rsync", "-avz", "--progress",
                "-e", f"ssh -o ControlPath={control_path}",
                src, local_dir + "/"
            ])

# Close SSH connection
subprocess.run(["ssh", "-S", control_path, "-O", "exit", ssh_target])