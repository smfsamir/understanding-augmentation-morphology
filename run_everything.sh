#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_everything.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_everything.error
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

# TODO: loop over everything with a range variable. feed range variable into the main_show_results.py. Resulting output: two files: all_results_form_{i} and all_results_form_{i}.csv. 
# LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")

# function run_one_iteration {
#     SEED="$1"
#     python main_grid_search.py $SEED --train_initial || echo "training initial models failed" && exit 1
#     python main_grid_search.py $SEED --train_uncertainty || echo "training uncertainty models failed" exit 1
#     python main_grid_search.py $SEED --train_random || echo "training random models failed" exit 1
#     python main_grid_search.py $SEED --train_uat || echo "training uat models failed"

#     # ## 
#     python main_evaluate_comp_gen.py $SEED || echo "generating compositional generalization results failed" && exit 1
#     python main_show_results.py $SEED --show_results || echo "Generating iid results failed" && exit 1
#     python main_show_results.py $SEED --show_results_compositional || echo "Generating comp. gen results failed" && exit 1

#     for lang in "${LANGUAGES[@]}"
#     do
#         rm -rf "/home/fsamir8/scratch/augmentation_subset_select/${lang}_${SEED}"
#     done
# }

srun --ntasks=1  ./run_one_iteration.sh 0 &
# srun --ntasks=1 ./run_one_iteration.sh 1 &
# srun --ntasks=1 ./run_one_iteration.sh 2 &
# srun --ntasks=1 ./run_one_iteration.sh 3 &
# srun --ntasks=1 ./run_one_iteration.sh 4 &
wait