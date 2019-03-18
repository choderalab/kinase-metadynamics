#!/usr/bin/env bash
# Set walltime limit
#BSUB -W 24:00
#
# Set output file
#BSUB -o  20ns_96.0_5UG9.log
#
# Specify node group
#BSUB -m "ls-gpu lt-gpu lp-gpu"
#BSUB -q gpuqueue
#
# nodes: number of nodes and GPU request (ptile: number of processes per node)
#BSUB -n 1 -R "rusage[mem=8] span[ptile=1]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#
# job name (default = name of script file)
#BSUB -J "20ns_96.0_5UG9"
#

module add cuda/9.0

python run_mtd.py
