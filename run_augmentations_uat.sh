#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --gpus=4
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_uat.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_uat.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

python main_grid_search.py --train_uat