#!/usr/bin/env python3
import subprocess
from pathlib import Path
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluations on all checkpoints in a directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--config', type=str, default='configs/eval/base.json',
                       help='Base config file to use')
    parser.add_argument('--additional_overrides', nargs='*', default=[],
                       help='Additional override arguments')
    return parser.parse_args()

def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
        
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Run evaluation for each checkpoint
    for checkpoint in checkpoints:
        print(f"\nEvaluating {checkpoint}")
        cmd = [
            "python", "visreps/run.py",
            "--mode=eval",
            "--config", args.config,
            "--override",
            f"checkpoint_path={checkpoint}",
            "load_model_from=checkpoint",
            *args.additional_overrides
        ]
        try:
            # Run with explicit process group and wait for completion
            process = subprocess.run(
                cmd,
                check=True,  # Raise exception on non-zero return code
                start_new_session=False,  # Don't create new process group
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(process.stdout)
            if process.stderr:
                print(f"Stderr: {process.stderr}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation for {checkpoint}: {e}", file=sys.stderr)
            print(f"Stderr: {e.stderr}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("\nInterrupted by user. Cleaning up...")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Cleaning up...")
        sys.exit(1) 