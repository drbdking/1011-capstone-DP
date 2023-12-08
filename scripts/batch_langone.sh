#!/bin/bash
for alpha in 0.5, 0.8, 1 
do
    for lr in 0.00001 0.000005
        do
        sbatch adv_fine_tune_langone.slurm $alpha $lr;
        echo "Submitted job for alpha = $alpha, lr = $lr"
done