#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/byt5_train.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/byt5_train.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

python main_byt5.py train-model fin