#!/bin/bash
adv_or_not=$1
alpha=0
for lr in 0.00001 0.000005
    do
        sbatch scripts/adv_fine_tune_langone.slurm $alpha $lr;
        echo "Submitted job for alpha = $alpha, lr = $lr"
    done
