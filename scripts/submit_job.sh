#!/bin/bash

# replace bsub with the scheduling command for the particular cluster
# first check if use_gpu is set to true
if [ "$5" = "true" ]; then
    # echo "Using GPU"
    bsub -n $1 -J "exps[1-${2}]" -o scripts/output.out \
                    -q $3 -gpu "num=1" "${4} jobid=\$LSB_JOBINDEX"
else
    # echo "Using CPU"
    bsub -n $1 -J "exps[1-${2}]" -o scripts/output.out \
                    -q $3 "${4} jobid=\$LSB_JOBINDEX"
fi