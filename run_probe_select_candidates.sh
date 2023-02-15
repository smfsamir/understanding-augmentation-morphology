#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_probe.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_probe.error
#SBATCH --mem=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fsamir@mail.ubc.ca

python main_probe_select_candidates.py