#!/bin/bash

for alpha in 0.8 1.0 1.5 2.0
do
    for adv_lr in 5e-4 1e-5
    do
        for embedding_mode in use_separate_embedding use_bert_embedding
        do
            sbatch scripts/adv_fine_tune_langone.slurm $alpha $adv_lr $embedding_mode;
            echo "Submitted job for alpha = $alpha, adv_lr = $adv_lr, $embedding_mode"
        done
    done
done
