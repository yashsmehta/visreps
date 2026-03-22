#!/usr/bin/env python3
import os, subprocess, sys, threading, time

# --- Config ---
# Each job specifies a folder, which subdirs to copy, and which files to grab.
jobs = [
    {
        "folder": "convnext_base_default",
        "subdirs": ["cfg1000a"],
        "files": ["checkpoint_epoch_20.pth", "config.json"],
    },
    {
        "folder": "convnext_base_clip_pca",
        "subdirs": ["cfg32a"],
        "files": ["checkpoint_epoch_20.pth", "config.json"],
    },
]
remote_base = "/scratch4/mbonner5/ymehta3/visreps/model_checkpoints"
local_base = "/data/ymehta3"


def fmt_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def ssh_cmd(*args):
    """Run a command over the persistent SSH connection."""
    return subprocess.run(
        ["ssh", "-o", f"ControlPath={control_path}", ssh_target, *args],
        capture_output=True, text=True,
    )


def get_remote_size(remote_path):
    """Get file size on remote via stat."""
    result = ssh_cmd(f"stat -c %s {remote_path} 2>/dev/null")
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


BAR_WIDTH = 25
BLOCKS = " ▏▎▍▌▋▊▉█"


def draw_progress(local_path, total, t0, done_event):
    """Live progress bar that polls local file size until done_event is set."""
    while not done_event.is_set():
        try:
            current = os.path.getsize(local_path)
        except FileNotFoundError:
            current = 0
        frac = min(current / total, 1.0) if total > 0 else 0
        elapsed = time.time() - t0
        speed = current / elapsed if elapsed > 0 else 0

        # Build bar with fractional block characters
        filled_exact = frac * BAR_WIDTH
        full = int(filled_exact)
        partial_idx = int((filled_exact - full) * (len(BLOCKS) - 1))
        bar = "█" * full
        if full < BAR_WIDTH:
            bar += BLOCKS[partial_idx]
            bar += " " * (BAR_WIDTH - full - 1)

        pct = frac * 100
        line = f"\r  │{bar}│ {pct:5.1f}%  {fmt_size(current):>9s}  {fmt_size(speed)}/s"
        sys.stdout.write(line)
        sys.stdout.flush()
        done_event.wait(timeout=0.2)


# --- SSH setup ---
host = "rockfish"
control_path = "/tmp/ssh_mux_%h"

print("Connecting to Rockfish...", end=" ", flush=True)
test = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "true"],
    capture_output=True,
)
if test.returncode == 0:
    ssh_target = host
else:
    print()
    user = input(
        "Keyless SSH failed. Enter your Rockfish username: "
    ).strip()
    ssh_target = f"{user}@{host}"

# Open persistent SSH connection
subprocess.run(
    [
        "ssh", "-MNf",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={control_path}",
        "-o", "ControlPersist=10m",
        ssh_target,
    ],
    capture_output=True,
)
print("connected.")

# --- Copy files ---
copied, skipped, failed = 0, 0, 0

for job in jobs:
    folder = job["folder"]
    remote_folder = f"{remote_base}/{folder}"
    ls_targets = " ".join(f"{remote_folder}/{s}" for s in job["subdirs"])

    print(f"\n{'─' * 50}")
    print(f"  {folder}")
    print(f"{'─' * 50}")

    result = subprocess.run(
        [
            "ssh", "-o", f"ControlPath={control_path}", ssh_target,
            f"ls -d {ls_targets} 2>/dev/null",
        ],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print(f"  No matching directories on remote. Skipping.")
        continue

    subdirs = result.stdout.split()

    for subdir in subdirs:
        cfg = os.path.basename(subdir)
        local_dir = f"{local_base}/{folder}/{cfg}"
        os.makedirs(local_dir, exist_ok=True)

        for f in job["files"]:
            local_path = os.path.join(local_dir, f)
            short_name = f"{cfg}/{f}"

            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                print(f"  skip  {short_name:45s} ({fmt_size(size)})")
                skipped += 1
                continue

            remote_path = f"{subdir}/{f}"
            src = f"{ssh_target}:{remote_path}"
            total = get_remote_size(remote_path)
            print(f"  copy  {short_name:40s} {fmt_size(total):>9s}")

            t0 = time.time()
            done_event = threading.Event()
            progress_thread = threading.Thread(
                target=draw_progress,
                args=(local_path, total, t0, done_event),
                daemon=True,
            )
            progress_thread.start()

            proc = subprocess.run(
                [
                    "rsync", "-az",
                    "-e", f"ssh -o ControlPath={control_path}",
                    src, local_dir + "/",
                ],
                capture_output=True, text=True,
            )
            done_event.set()
            progress_thread.join()
            elapsed = time.time() - t0

            if proc.returncode != 0:
                sys.stdout.write(f"\r  {'FAILED':>{BAR_WIDTH + 40}}\n")
                stderr = proc.stderr.strip()
                if stderr:
                    print(f"        {stderr.splitlines()[0]}")
                failed += 1
            else:
                final_size = os.path.getsize(local_path) if os.path.exists(local_path) else total
                speed = final_size / elapsed if elapsed > 0 else 0
                bar = "█" * BAR_WIDTH
                sys.stdout.write(
                    f"\r  │{bar}│ 100.0%  {fmt_size(final_size):>9s}  {fmt_size(speed)}/s  {fmt_time(elapsed)}\n"
                )
                copied += 1

# Close SSH connection
subprocess.run(
    ["ssh", "-S", control_path, "-O", "exit", ssh_target],
    capture_output=True,
)

# --- Summary ---
print(f"\n{'─' * 50}")
parts = []
if copied:
    parts.append(f"{copied} copied")
if skipped:
    parts.append(f"{skipped} skipped")
if failed:
    parts.append(f"{failed} failed")
print(f"  Done: {', '.join(parts) if parts else 'nothing to do'}.")
print(f"{'─' * 50}")
