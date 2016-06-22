#!/bin/bash

# echo on
set -x

# number of runs to perform of each experiment
N_RUNS=1

# train deep chroma model
for i in `seq $N_RUNS`
do
    python -m chordrec.chroma with deep_chroma.yaml
done

