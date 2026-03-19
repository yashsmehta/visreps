#!/usr/bin/env python3
import os, subprocess

# --- Config ---
# Each job specifies a folder, which subdirs to copy, and which files to grab.
# Epoch 0 (untrained network) is the same across all label granularities,
# so we only need it for the default (1000-way) network.
jobs = [
    {
        "folder": "resnet50_clip_pca",
        "subdirs": [f"cfg{n}{s}" for n in [8, 32] for s in "a"],
        "files": ["checkpoint_epoch_70.pth", "config.json"],
    },
]
remote_base = "/scratch4/mbonner5/ymehta3/visreps/model_checkpoints"
local_base = "/data/ymehta3"

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
for job in jobs:
    folder = job["folder"]
    remote_folder = f"{remote_base}/{folder}"
    ls_targets = " ".join(f"{remote_folder}/{s}" for s in job["subdirs"])
    result = subprocess.run(
        ["ssh", "-o", f"ControlPath={control_path}", ssh_target,
         f"ls -d {ls_targets} 2>/dev/null"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        print(f"⚠ No matching directories found for {folder} on remote. Skipping.")
        continue
    subdirs = result.stdout.split()

    for subdir in subdirs:
        cfg = os.path.basename(subdir)
        local_dir = f"{local_base}/{folder}/{cfg}"
        os.makedirs(local_dir, exist_ok=True)
        for f in job["files"]:
            local_path = os.path.join(local_dir, f)
            if os.path.exists(local_path):
                print(f"Skipping {local_path} (already exists)")
                continue
            src = f"{ssh_target}:{subdir}/{f}"
            print(f"Copying {src} → {local_dir}")
            subprocess.run([
                "rsync", "-avz", "--progress",
                "-e", f"ssh -o ControlPath={control_path}",
                src, local_dir + "/"
            ])

# Close SSH connection
subprocess.run(["ssh", "-S", control_path, "-O", "exit", ssh_target])