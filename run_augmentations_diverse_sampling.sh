#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_random.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_random.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

LANGUAGES=("bengali" "turkish" "finnish" "georgian" "arabic" "navajo" "spanish")
DATA_SIZES=(128 256 512 1024 2048)
SCRATCH_PATH="/home/fsamir8/scratch/augmentation_subset_select"
for lang in "${LANGUAGES[@]}"
do
	    for data_size in "${DATA_SIZES[@]}" 
	    do
            # TODO: make sure u change this!!
            RESULTS_FILE="$SCRATCH_PATH/$lang/diversity_sample_num_aug=${data_size}_k=32/final_results.txt"
            if [ -f "$RESULTS_FILE" ]; then
                echo "Model trained already for $lang with $data_size augmented datapoints."
            else  
                echo "Commencing training for $lang with $data_size augmented datapoints"
                python main_transformer.py $lang diversity_sample $data_size --run_k_diverse_sampling_pipeline
            fi
	    done 
done
