import os
import argparse
import subprocess
import glob
import re
import sys

# --- Hardcoded list of target epochs to process ---
TARGET_EPOCHS = [5, 10] # Example: process only epochs 5, and 10
# --------------------------------------------------

def run_command(command, print_command=False):
    """Runs a command and prints limited output/errors."""
    if print_command: # Optionally print command if needed for debugging
        print(f"---> Running: {" ".join(command)}") 
    try:
        # Run quietly unless there's an error
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        # Print details only on error
        print(f"Error running command: {" ".join(command)}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print("Stderr:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("Stdout:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("--------------------", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"Error: Command not found (make sure scripts are in PATH or use full paths): {command[0]}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run feature extraction and eigenspectra analysis pipeline for models in a directory structure.")
    parser.add_argument('--base_dir', type=str, default='model_checkpoints/imagenet_pca',
                        help='Base directory containing experiment folders (e.g., model_checkpoints/imagenet_pca)')
    parser.add_argument('--dataset', type=str, default='imagenet-mini-50', # Default might need adjustment based on experiments
                        choices=['tiny-imagenet', 'imagenet', 'imagenet-mini-50'],
                        help='Dataset used for the experiments (needed for extract_representations.py)')
    parser.add_argument('--python_executable', type=str, default=sys.executable,
                        help="Path to the python executable to use for running sub-scripts.")
    parser.add_argument('--extract_script', type=str, default='visreps/analysis/extract_representations.py',
                        help="Path to the extract_representations.py script.")
    parser.add_argument('--eigen_script', type=str, default='visreps/analysis/compute_eigenspectra.py',
                        help="Path to the compute_eigenspectra.py script.")

    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    # Derive experiment name from base_dir
    exp_name_from_base = os.path.basename(os.path.normpath(args.base_dir))

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(args.base_dir, 
                                      "*", 
                                      "checkpoint*.pth")
    all_checkpoint_files = glob.glob(checkpoint_pattern)

    if not all_checkpoint_files:
        print(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
        sys.exit(0)

    # --- Filter checkpoints by target epochs --- 
    targeted_checkpoints = []
    skipped_initial_count = 0
    for ckpt_path in all_checkpoint_files:
        epoch_match = re.search(r"epoch_(\d+)", os.path.basename(ckpt_path))
        if epoch_match:
            epoch_number = int(epoch_match.group(1))
            if epoch_number in TARGET_EPOCHS:
                targeted_checkpoints.append(ckpt_path)
            else:
                 skipped_initial_count += 1
        else:
            # Keep warning for unparseable names found during filtering
            print(f"Warning: Could not extract epoch number from checkpoint filename {os.path.basename(ckpt_path)}. Skipping.")
            # Consider if this should count towards failure_count later if needed

    if not targeted_checkpoints:
        print(f"Found {len(all_checkpoint_files)} checkpoints, but none matched the target epochs: {TARGET_EPOCHS}")
        sys.exit(0)
    
    print(f"Found {len(all_checkpoint_files)} checkpoints. Processing {len(targeted_checkpoints)} matching target epochs: {TARGET_EPOCHS}")
    # -------------------------------------------

    # Sort the targeted checkpoints alphabetically by path
    targeted_checkpoints.sort()

    success_count = 0
    failure_count = 0
    # skipped_count now represents failures during processing, not non-targeted epochs

    total_to_process = len(targeted_checkpoints)
    for i, ckpt_path in enumerate(targeted_checkpoints):
        
        print(f"\nProcessing file {i+1}/{total_to_process}: {ckpt_path}") 

        try:
            # Extract parts from path: base_dir/<cfg_dir>/<ckpt_file>
            rel_path = os.path.relpath(ckpt_path, args.base_dir)
            parts = rel_path.split(os.sep)
            if len(parts) != 2:
                print(f"  Warning: Skipping unexpected path structure (expected base_dir/cfgXX/ckpt.pth): {ckpt_path}")
                failure_count += 1
                continue
            cfg_dir = parts[0]
            ckpt_file = parts[1]

            cfg_match = re.match(r"cfg(\d+)", cfg_dir)
            if not cfg_match:
                print(f"  Warning: Could not parse cfg_id from directory {cfg_dir}. Skipping {ckpt_path}")
                failure_count += 1
                continue
            cfg_id = cfg_match.group(1)
            
            epoch_match = re.search(r"epoch_(\d+)", ckpt_file)
            epoch_number = int(epoch_match.group(1)) # Already checked it matches target epochs

            checkpoint_base_dir = os.path.dirname(ckpt_path)
            model_reps_dir = os.path.join(checkpoint_base_dir, "model_representations")
            expected_npz_filename = f"model_reps_epoch_{epoch_number}.npz"
            expected_npz_path = os.path.join(model_reps_dir, expected_npz_filename)

            # Create model_representations directory only if epoch is targeted
            os.makedirs(model_reps_dir, exist_ok=True)

        except Exception as e:
            print(f"  Error parsing path {ckpt_path}: {e}", file=sys.stderr)
            failure_count += 1
            continue
        
        # --- Step 1: Run extract_representations.py --- 
        print("  Running representation extraction...")
        extract_cmd = [
            args.python_executable, args.extract_script,
            "--load_from", "checkpoint",
            "--exp_name", exp_name_from_base, 
            "--cfg_id", str(cfg_id),
            "--checkpoint_model", ckpt_file,
            "--dataset", args.dataset 
        ]
        if not run_command(extract_cmd):
            print(f"  Failed to extract representations for {ckpt_path}. Skipping subsequent steps.")
            failure_count += 1
            continue

        # Check if the expected output file was created
        if not os.path.exists(expected_npz_path):
             print(f"  Error: Expected output file not found after running extraction: {expected_npz_path}", file=sys.stderr)
             print("  Please check the output of the extraction script.")
             failure_count += 1
             continue

        # --- Step 2: Run compute_eigenspectra.py --- 
        print("  Computing eigenspectra...")
        eigen_cmd = [
            args.python_executable, args.eigen_script,
            "--input_path", expected_npz_path
        ]
        if not run_command(eigen_cmd):
            print(f"  Failed to compute eigenspectra for {expected_npz_path}.")
            failure_count += 1
            continue
        else:
            try:
                os.remove(expected_npz_path)
                print(f"  Deleted intermediate representations.") 
            except OSError as e:
                print(f"  Warning: Failed to delete intermediate file {expected_npz_path}: {e}", file=sys.stderr)

        success_count += 1

    print(f"\n=== Pipeline Summary ===")
    print(f"Total checkpoints found: {len(all_checkpoint_files)}")
    print(f"Target Epochs: {TARGET_EPOCHS}")
    print(f"Checkpoints processed: {success_count}")
    print(f"Skipped (epoch not targeted): {skipped_initial_count}")
    print(f"Failed/Skipped (processing errors): {failure_count}")
    print("=======================")

if __name__ == "__main__":
    main() 