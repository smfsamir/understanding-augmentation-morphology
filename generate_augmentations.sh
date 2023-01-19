#!/bin/bash

# NOTE: this works!! just need to iterate over every language.
DATA_PATH="/home/fsamir8/projects/rrg-msilfver/fsamir8/conll2018/task1/all/"
# iterate through all files
    # then, if it's a low-train, use it.
    # TODO: just do it for the target languages
    # TODO: then, do it for 
LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")

for fname_full in "${DATA_PATH}"/*
do
    fname=$(basename "$fname_full")
    if [[ $fname == *"train-low" ]]; then
        python augment.py $DATA_PATH "$fname" --examples 10000
    fi
done
