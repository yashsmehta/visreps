#!/usr/bin/env bash
PARTITION="v100"

# Get list of nodes in the partition
mapfile -t NODES < <(sinfo -h -N -p "$PARTITION" -o "%N")

TOTAL_GPUS=0
ALLOCATED_GPUS=0

for NODE in "${NODES[@]}"; do
  # Extract total GPUs per node from Gres
  GPUS_NODE=$(scontrol show node "$NODE" | grep -Po 'Gres=gpu:[^:]+:\K[0-9]+' || echo 0)
  TOTAL_GPUS=$((TOTAL_GPUS + GPUS_NODE))
  # Extract allocated GPUs per node from AllocTRES
  ALLOC=$(scontrol show node "$NODE" | grep -Po 'gres/gpu=\K[0-9]+' | head -1 || echo 0)
  ALLOCATED_GPUS=$((ALLOCATED_GPUS + ALLOC))
done

FREE_GPUS=$((TOTAL_GPUS - ALLOCATED_GPUS))
USERS=$(squeue -h -p "$PARTITION" -o "%u" | sort -u | wc -l)
JOBS=$(squeue -h -p "$PARTITION" | wc -l)

echo "Partition: $PARTITION"
echo "Total GPUs: $TOTAL_GPUS"
echo "Allocated GPUs: $ALLOCATED_GPUS"
echo "Free GPUs: $FREE_GPUS"
echo "Unique Users: $USERS"
echo "Total Jobs: $JOBS"

echo ""
echo "Top 5 Users by Job Count and GPU Usage (% of total GPUs):"
# Table header
printf "%-12s | %5s | %5s | %6s\n" "USER" "JOBS" "GPUS" "%TOTAL"
echo "--------------+-------+-------+----------"
# Data rows
squeue -h -p "$PARTITION" -o "%u %b" | \
awk '{ user=$1; gp=0; if(match($2, /gpu(:[^:]+)?:([0-9]+)/, m)) gp=m[2]; jobs[user]++; gpus[user]+=gp } END { for (u in jobs) print u, jobs[u], gpus[u] }' | \
sort -k2,2nr | head -n 5 | \
while read user jobs gpus; do
  percent=$(awk -v gpus=$gpus -v total=$TOTAL_GPUS 'BEGIN {printf "%.1f", gpus*100/total}')
  printf "%-12s | %5d | %5d | %5.1f%%\n" "$user" "$jobs" "$gpus" "$percent"
done 