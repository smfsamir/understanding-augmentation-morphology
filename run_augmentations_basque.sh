#!/bin/bash
#SBATCH --time=13:30:00
#SBATCH --gpus=1
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_basque.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_basque.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

python main_transformer.py basque initial --run_initial_pipeline # the initial keyword doesn't matter
python main_transformer.py basque uncertainty_sample --run_uncertainty_sampling_pipeline # the initial keyword doesn't matter
python main_transformer.py basque diversity_sample --run_k_diverse_sampling_pipeline # the augmentation strategy keyword doesn't matter since we're using a pipeline.
python main_transformer.py basque random --run_random_sampling_pipeline # the augmentation strategy keyword doesn't matter, since we're using a pipeline.