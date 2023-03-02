#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:2
#SBATCH --array=0-5
#SBATCH --mem-per-cpu=10G
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_everything.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_everything.error
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")
SEED="$SLURM_ARRAY_TASK_ID"
echo "Running everything with seed $SEED"
python main_grid_search.py $SEED 100000 --train_initial || { echo "training initial models failed" ; exit 1; }
# python main_grid_search.py $SEED --train_uncertainty || { echo "training uncertainty models failed" ; exit 1; }
python main_grid_search.py $SEED 100000 --train_random || { echo "training random models failed" ; exit 1; }
python main_grid_search.py $SEED 100000 --train_uat || { echo "training uat models failed" ; exit 1; }

# ## 
python main_evaluate_comp_gen.py $SEED 100000 || { echo "generating compositional generalization results failed" ; exit 1; }
python main_show_results.py $SEED 100000 --show_results || { echo "Generating iid results failed" ; exit 1; }
python main_show_results.py $SEED 100000 --show_results_compositional || { echo "Generating comp. gen results failed" ; exit 1; }

for lang in "${LANGUAGES[@]}"
do
    rm -rf "/home/fsamir8/scratch/augmentation_subset_select/${lang}_seed=${SEED}_aug_pool_size=100000" 
done