#!/bin/bash

# echo on
set -x

# number of runs to perform of each experiment
N_RUNS=10

# run chord classification ...
# ... with deep chroma extractor
for i in `seq $N_RUNS`
do
    python -m chordrec.chroma with deep_chroma.yaml
done

# ... with simple chromas
for i in `seq $N_RUNS`
do
    python -m chordrec.classify with chroma.yaml
done

# ... with weighted, logarithmised chromas
for i in `seq $N_RUNS`
do
    python -m chordrec.classify with chroma_wlog.yaml
done

# ... with logarithmic filtered spectrogram
for i in `seq $N_RUNS`
do
    python -m chordrec.classify with logfiltspec.yaml
done
