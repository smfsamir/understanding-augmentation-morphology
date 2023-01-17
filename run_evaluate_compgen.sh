#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus=3
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_compgen.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_compgen.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

python main_evaluate_comp_gen.py