#!/bin/bash

lr="1e-3"
train_epochs=3000
data_names=(
 "cub-2view" "MSRCV1-3view" "HW-3view"
 "Youtube-6view" "OutdoorScene-4view" "RGB-D"
 "nuswide-2view-s" "xrmb-2view-s" "xmedia-2view-s"
)
alphas=(-5 -4 -3 -2 -1 0 1 2 3 4 5)

for data_name in "${data_names[@]}"; do
    for alpha in "${alphas[@]}"; do
        python ./main.py --data_name "$data_name" --lr "$lr" --train_epochs "$train_epochs" --alpha "$alpha"
    done
done