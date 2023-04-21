#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/byt5_train_all.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/byt5_train_all.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

module load gcc/9.3.0 arrow python scipy-stack

python main_byt5.py train-model --construct_arrow_dataset
# python main_byt5.py test-model 