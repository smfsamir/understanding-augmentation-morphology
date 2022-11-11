#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

# rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/uncertainty_sample/

# python main_transformer.py bengali uncertainty_sample --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/random/

python main_transformer.py bengali random --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
python main_transformer.py bengali random --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
python main_transformer.py bengali random --train_model || { echo 'training model failed' ; exit 1; }
python main_transformer.py bengali random --generate || { echo 'generating predictions failed' ; exit 1; }
python main_transformer.py bengali random --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/uncertainty_sample/

# python main_transformer.py bengali uncertainty_sample --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }