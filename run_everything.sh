#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gpus=4
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_initial.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_initial.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

# TODO: loop over everything with a range variable. feed range variable into the main_show_results.py. Resulting output: two files: all_results_form_{i} and all_results_form_{i}.csv. 
LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")
for lang in "${LANGUAGES[@]}"
do
    # ls -d /home/fsamir8/scratch/augmentation_subset_select/$lang/* 
    ls -d /home/fsamir8/scratch/augmentation_subset_select/$lang/* | xargs rm -r
    ## create models
    python main_grid_search.py --train_initial || echo "training initial models failed" && exit 1
    python main_grid_search.py --train_uncertainty || echo "training uncertainty models failed" exit 1
    python main_grid_search.py --train_random || echo "training random models failed" exit 1
    python main_grid_search.py --train_uat || echo "training uat models failed"

    ## 
    python main_evaluate_comp_gen.py || echo "generating compositional generalization results failed" && exit 1
    python main_show_results.py --show_results || echo "Generating iid results failed" && exit 1
    python main_show_results.py --show_results_compositional || echo "Generating comp. gen results failed" && exit 1
done