#!/bin/bash

#SBATCH --account=def-msilfver
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --time 01:30:00
#SBATCH --job-name=TrainInitialModel
#SBATCH --output=/home/fsamir8/scratch/augmentation_subset_select/bengali/initial/bengali.out
#SBATCH --error=/home/fsamir8/scratch/augmentation_subset_select/bengali/initial/bengali.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fsamir@mail.ubc.ca

python main_transformer.py bengali initial --train_model