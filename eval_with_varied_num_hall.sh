#!/bin/bash

: >results_varied_hall.txt
END=13

for ((i=4; i<=13; i++)); do
    echo "Training with $((2**i)) augmented examples" >> results_varied_hall.txt
    python main_train.py $((2**i)) >> results_varied_hall.txt
done