#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --gpus=1
#SBATCH --account=rrg-mageed
#SBATCH --account=rrg-mageed
#SBATCH --output=/scratch/fsamir8/augmentation_subset_select/augmentation_arabic.out
#SBATCH --error=/scratch/fsamir8/augmentation_subset_select/augmentation_arabic.error
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=fsamir@mail.ubc.ca

# python main_transformer.py turkish initial --run_initial_pipeline # the initial keyword doesn't matter
# python main_transformer.py turkish uncertainty_sample --run_uncertainty_sampling_pipeline # the initial keyword doesn't matter
# python main_transformer.py turkish diversity_sample --run_k_diverse_sampling_pipeline # the augmentation strategy keyword doesn't matter since we're using a pipeline.
# python main_transformer.py turkish random --run_random_sampling_pipeline # the augmentation strategy keyword doesn't matter, since we're using a pipeline.
python main_transformer.py turkish random 10000 --run_random_sampling_pipeline

# rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/uncertainty_sample/
# python main_transformer.py arabic uncertainty_sample --run_uncertainty_sampling_pipeline # the augmentation strategy keyword doesn't matter, since we're using a pipeline.

# python main_transformer.py arabic initial --prep_preproc_fairseq_data_initial || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py arabic initial --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py arabic initial --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py arabic initial --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py arabic initial --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# INITIAL_SCRATCH_PATH="/home/fsamir8/scratch/augmentation_subset_select/arabic/initial/"
# mv "${INITIAL_SCRATCH_PATH}/arabic_ids.pickle" "${INITIAL_SCRATCH_PATH}/arabic_ids_true_initial.pickle"
# mv "${INITIAL_SCRATCH_PATH}/arabic_src_dict.pickle" "${INITIAL_SCRATCH_PATH}/arabic_src_dict_true_initial.pickle"
# mv "${INITIAL_SCRATCH_PATH}/arabic_src_tokens.pickle" "${INITIAL_SCRATCH_PATH}/arabic_src_tokens_true_initial.pickle"
# mv "${INITIAL_SCRATCH_PATH}/arabic_embeddings.pickle" "${INITIAL_SCRATCH_PATH}/arabic_embeddings_true_initial.pickle"

# python main_transformer.py arabic uncertainty_sample --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py arabic uncertainty_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py arabic uncertainty_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py arabic uncertainty_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py arabic uncertainty_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# python main_transformer.py arabic random --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py arabic random --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py arabic random --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py arabic random --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py arabic random --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# python main_transformer.py turkish initial --prep_preproc_fairseq_data_initial || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py turkish initial --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py turkish initial --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py turkish initial --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py turkish initial --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# python main_transformer.py bengali uncertainty_sample --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# python main_transformer.py turkish uncertainty_sample --extract_log_likelihoods || { echo 'extracting log likelihoods failed' ; exit 1; }
# python main_transformer.py turkish random --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py turkish random --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py turkish random --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py turkish random --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py turkish random --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/random/
# python main_transformer.py turkish uncertainty_sample --extract_log_likelihoods || { echo 'extracting log likelihoods failed' ; exit 1; }
# python main_transformer.py turkish uncertainty_sample --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py turkish uncertainty_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py turkish uncertainty_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py turkish uncertainty_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py turkish uncertainty_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# python main_transformer.py bengali random --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py bengali random --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py bengali random --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py bengali random --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py bengali random --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/uncertainty_sample/

# python main_transformer.py bengali uncertainty_sample --prep_preproc_fairseq_data_augment || { echo 'creating src tgt files failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py bengali uncertainty_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }

# rm -rf /home/fsamir8/scratch/augmentation_subset_select/bengali/diversity_sample/
# python main_transformer.py bengali diversity_sample --prep_preproc_fairseq_data_augment 
# python main_transformer.py bengali diversity_sample --run_fairseq_binarizer || { echo 'running fairseq preprocesser failed' ; exit 1; }
# python main_transformer.py bengali diversity_sample --train_model || { echo 'training model failed' ; exit 1; }
# python main_transformer.py bengali diversity_sample --generate || { echo 'generating predictions failed' ; exit 1; }
# python main_transformer.py bengali diversity_sample --report_accuracy || { echo 'Reporting accuracy failed'; exit 1; }