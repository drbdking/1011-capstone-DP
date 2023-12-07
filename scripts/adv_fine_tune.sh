#!/bin/bash
for lr in 0.00005 0.00001 0.000005
do
    echo "-------------------- fine-tuning, learning rate = $lr ---------------------";
    python fine_tune.py --downsample 0.25 --num_epochs 5 --learning_rate $lr
done