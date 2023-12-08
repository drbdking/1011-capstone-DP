#!/bin/bash
for alpha in 1 1.5 
do
    echo "-------------------- fine-tuning, alpha = $alpha ---------------------";
    python adv_ft.py --downsample 0.25 --adv_mode 0 --num_epochs 5 --alpha $alpha --batch_size 64
done