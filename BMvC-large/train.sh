#!/bin/bash

lr="1e-3"
train_epochs=3000
data_names=(
 "nuswide-2view"
  "xrmb-2view"
  "xmedia-2view"
)
alphas=(-5 -4 -3 -2 -1 0 1 2 3 4 5)

for data_name in "${data_names[@]}"; do
    for alpha in "${alphas[@]}"; do
        python ./main.py --data_name "$data_name" --lr "$lr" --train_epochs "$train_epochs" --alpha "$alpha"
    done
done