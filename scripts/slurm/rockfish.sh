#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# rockfish.sh — Submit and monitor Slurm jobs on Rockfish
#               from the local machine over SSH.
#
# Usage:
#   ./scripts/slurm/rockfish.sh submit                 Push, pull, submit train jobs
#   ./scripts/slurm/rockfish.sh jobs                   Show running/pending jobs
#   ./scripts/slurm/rockfish.sh history [N]            Last N completed jobs (default 20)
#   ./scripts/slurm/rockfish.sh log <job_id>           Tail the output log
#   ./scripts/slurm/rockfish.sh err <job_id>           Tail the error log
#   ./scripts/slurm/rockfish.sh cancel <job_id>        Cancel a job (or "all")
# ─────────────────────────────────────────────────────────────

set -euo pipefail

REMOTE="rockfish"
REMOTE_REPO="/scratch4/mbonner5/ymehta3/visreps"
REMOTE_LOGS="$REMOTE_REPO/scripts/slurm/slurm_logs"
LOCAL_REPO="$(cd "$(dirname "$0")/../.." && pwd)"

# ── Helpers ──────────────────────────────────────────────────

ssh_rf() { ssh "$REMOTE" "$@"; }

die() { echo "Error: $*" >&2; exit 1; }

SLURM_FILES=(
    "scripts/slurm/"
    "configs/train/"
)

ensure_synced() {
    # Auto-commit only Slurm-related files, then push + pull on Rockfish
    local changed
    changed="$(git -C "$LOCAL_REPO" diff --name-only -- "${SLURM_FILES[@]}")"
    if [ -n "$changed" ]; then
        echo "── Committing Slurm-related changes..."
        git -C "$LOCAL_REPO" add -- "${SLURM_FILES[@]}"
        git -C "$LOCAL_REPO" commit -m "🔧 Update Slurm configs for submission"
        echo ""
    fi

    echo "── Pushing to remote..."
    git -C "$LOCAL_REPO" push || die "git push failed"

    echo "── Pulling on Rockfish..."
    ssh_rf "cd $REMOTE_REPO && git pull --ff-only" || die "git pull failed on Rockfish"
    echo ""
}

# ── Commands ─────────────────────────────────────────────────

cmd_submit() {
    ensure_synced

    # Dry-run to enumerate jobs (parses TOTAL=N from train_scheduler.py --dry-run)
    echo "── Checking jobs to submit..."
    local dry_output
    dry_output="$(ssh_rf "cd $REMOTE_REPO && source .venv/bin/activate && python scripts/slurm/train_scheduler.py --dry-run")"

    local total
    total="$(echo "$dry_output" | grep '^TOTAL=' | cut -d= -f2)"

    if [ -z "$total" ] || [ "$total" -eq 0 ]; then
        echo "No jobs to submit."
        return
    fi

    echo ""
    echo "$dry_output" | grep -v '^TOTAL='
    echo ""
    read -rp "Submit $total job(s) to Rockfish? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        return
    fi

    echo ""
    echo "── Submitting training jobs..."
    ssh_rf "cd $REMOTE_REPO && source .venv/bin/activate && python scripts/slurm/train_scheduler.py"
}

cmd_jobs() {
    # Fetch raw squeue data (skip header), parse job names like vr_RN50_c16
    local raw
    raw="$(ssh_rf "squeue -u \$USER -o '%i|%j|%t|%M|%R' --noheader --sort=-i" 2>/dev/null)"

    if [ -z "$raw" ]; then
        echo "No active jobs."
        return
    fi

    printf "── Active Slurm jobs ──────────────────────────────────────\n"
    printf "  %-10s  %-10s  %-8s  %-4s  %-10s  %s\n" \
           "JOB ID" "MODEL" "CLASSES" "ST" "TIME" "NODE"
    printf "  %-10s  %-10s  %-8s  %-4s  %-10s  %s\n" \
           "──────" "─────" "───────" "──" "────" "────"

    while IFS='|' read -r jid name state elapsed reason; do
        # Parse vr_{tag}_{suffix} job names (tags set by train_scheduler.py)
        if [[ "$name" =~ ^vr_([^_]+)_(.+)$ ]]; then
            model="${BASH_REMATCH[1]}"
            suffix="${BASH_REMATCH[2]}"
            if [[ "$suffix" == "std" ]]; then
                classes="1000"
            elif [[ "$suffix" =~ ^c([0-9]+)$ ]]; then
                classes="${BASH_REMATCH[1]}-way"
            else
                classes="$suffix"
            fi
        else
            model="$name"
            classes="—"
        fi
        printf "  %-10s  %-10s  %-8s  %-4s  %-10s  %s\n" \
               "$jid" "$model" "$classes" "$state" "$elapsed" "$reason"
    done <<< "$raw"

    local count
    count="$(echo "$raw" | wc -l)"
    printf "──────────────────────────────────────────────────────────\n"
    printf "  %s job(s) active\n" "$count"
}

cmd_history() {
    local n="${1:-20}"
    echo "── Last $n finished jobs ──"
    ssh_rf "sacct -u \$USER -n --format=JobID%-12,JobName%-20,State%-12,Elapsed%-10,End%-20 \
            | grep -v '\.batch' | grep -v '\.extern' | tail -n $n"
}

cmd_log() {
    local job_id="${1:?Usage: rockfish.sh log <job_id>}"
    echo "── Output log for job $job_id ──"
    ssh_rf "tail -n 80 $REMOTE_LOGS/${job_id}.out 2>/dev/null || echo 'Log not found: ${job_id}.out'"
}

cmd_err() {
    local job_id="${1:?Usage: rockfish.sh err <job_id>}"
    echo "── Error log for job $job_id ──"
    ssh_rf "tail -n 80 $REMOTE_LOGS/${job_id}.err 2>/dev/null || echo 'Log not found: ${job_id}.err'"
}

cmd_cancel() {
    local job_id="${1:?Usage: rockfish.sh cancel <job_id|all>}"
    if [ "$job_id" = "all" ]; then
        echo "Cancelling all your jobs..."
        ssh_rf "scancel -u \$USER"
    else
        echo "Cancelling job $job_id..."
        ssh_rf "scancel $job_id"
    fi
    echo "Done."
}

# ── Dispatch ─────────────────────────────────────────────────

cmd="${1:-}"
shift || true

case "$cmd" in
    submit)  cmd_submit ;;
    jobs)    cmd_jobs ;;
    history) cmd_history "$@" ;;
    log)     cmd_log "$@" ;;
    err)     cmd_err "$@" ;;
    cancel)  cmd_cancel "$@" ;;
    *)
        echo "Usage: rockfish.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  submit                Push, pull on Rockfish, submit train jobs"
        echo "  jobs                  Show running/pending jobs"
        echo "  history [N]           Last N completed jobs (default 20)"
        echo "  log <job_id>          Tail stdout log"
        echo "  err <job_id>          Tail stderr log"
        echo "  cancel <job_id|all>   Cancel job(s)"
        exit 1
        ;;
esac
