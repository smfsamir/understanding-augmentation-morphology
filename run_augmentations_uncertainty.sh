#!/bin/bash
#SBATCH --time=40:00:00
#SBATCH --gpus=1
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_uncertainty.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_uncertainty.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

# LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")
# R=(1.0 0.75)
# USE_SOFTMAX=("" "--use_softmax_normalizer")
# DATA_SIZES=(128 256 512 1024 2048)
# for lang in "${LANGUAGES[@]}"
# do
#     for r in "${R[@]}"
#     do 
#         for use_softmax in "${USE_SOFTMAX[@]}"
# 	do 
# 	    for data_size in "${DATA_SIZES[@]}" 
# 	    do
#             # TODO: write the if-statement file check here as well.
#                 python main_transformer.py $lang uncertainty_sample $data_size --run_uncertainty_sampling_pipeline  --r $r $use_softmax 
# 	    done 
# 	done 
#     done 
# done
python main_grid_search.py --train_uncertainty