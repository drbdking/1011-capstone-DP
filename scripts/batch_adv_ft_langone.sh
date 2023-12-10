#!/bin/bash

for alpha in 2.0 5.0 8.0
do
    for adv_lr in 5e-4 5e-5
    do
        sbatch scripts/adv_fine_tune_langone.slurm $alpha $adv_lr;
        echo "Submitted job for alpha = $alpha, adv_lr = $adv_lr"
    done
done
