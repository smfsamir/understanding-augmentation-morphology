#!/bin/bash

LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")
SEED="$1"
echo "Running everything with seed $SEED"
python main_grid_search.py $SEED --train_initial || { echo "training initial models failed" ; exit 1; }
python main_grid_search.py $SEED --train_uncertainty || { echo "training uncertainty models failed" ; exit 1; }
python main_grid_search.py $SEED --train_random || { echo "training random models failed" ; exit 1; }
python main_grid_search.py $SEED --train_uat || { echo "training uat models failed" ; exit 1; }

# ## 
python main_evaluate_comp_gen.py $SEED || { echo "generating compositional generalization results failed" ; exit 1; }
python main_show_results.py $SEED --show_results || { echo "Generating iid results failed" ; exit 1; }
python main_show_results.py $SEED --show_results_compositional || { echo "Generating comp. gen results failed" ; exit 1; }

for lang in "${LANGUAGES[@]}"
do
    rm -rf "/home/fsamir8/scratch/augmentation_subset_select/${lang}_seed=${SEED}"
done
