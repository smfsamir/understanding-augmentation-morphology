import numpy as np
import pickle
import pdb
import subprocess
import os 
import argparse
import pandas as pd
from collections import defaultdict

from packages.utils.constants import SIGM_DATA_PATH, SCRATCH_PATH, FAIRSEQ_SCRIPTS_PATH, INITIAL_MODEL_PARAMS
from packages.augmentation.select_highest_loss import HighLossSampler
from packages.augmentation.subset_selecter_strategy import get_subset_selecter
from packages.fairseq_utils.dataloading_utils import get_initial_generation_frame, get_augmentation_example_lengths
from packages.utils.util_functions import get_model_augment_path, load_gold_train_validation_test, tokenize_row_src, tokenize_row_tgt

def _write_split(language, augmentation_type, split_frame, split_name, **kwargs): 
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    if not os.path.exists(model_augment_path):
        os.makedirs(model_augment_path)

    # write src
    with open(f"{model_augment_path}/{language}-{split_name}.src", "w") as fseq_src_f:
        split_frame.apply(lambda row: fseq_src_f.write(f"{tokenize_row_src(row)}\n"), 
                        axis=1)  # rows
    # write tgt
    with open(f"{model_augment_path}/{language}-{split_name}.tgt", "w") as fseq_tgt_f:
        split_frame.apply(lambda row: fseq_tgt_f.write(f"{tokenize_row_tgt(row)}\n"), 
                        axis=1)  # rows


def prep_preproc_fairseq_data_initial(language: str, augmentation_type: str, **kwargs):
    train_medium = kwargs['train_medium']

    # TODO: would add the copies here.
    train_frame, validation_frame, test_frame = load_gold_train_validation_test(language, train_medium)

    test_frame = get_initial_generation_frame(language, kwargs['aug_pool_size'])

    _write_split(language, augmentation_type, train_frame, "train", **kwargs)
    _write_split(language, augmentation_type, validation_frame, "valid", **kwargs)
    _write_split(language, augmentation_type, test_frame, "test", **kwargs)

def run_fairseq_binarizer(language, augmentation_type, **kwargs):
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/preprocess.sh", model_augment_path, language])
    print(f"Obtained {result} result")

def train_model(language, augmentation_type, **kwargs):
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    algorithm_type = model_augment_path.split('/')[-1]
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", model_augment_path, language, algorithm_type, str(kwargs['rand_seed'])])
    print(f"Obtained {result} result")

def generate(language, augmentation_type, **kwargs):
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    algorithm_type = model_augment_path.split('/')[-1]
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/generate.sh", model_augment_path, language, algorithm_type])
    print(f"Obtained {result} result")

# TODO: delete this later; use from the util_functions module.
def get_number_test_examples(language, **kwargs):
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    return len(test_frame)

def report_accuracy(language, augmentation_type, num_test_examples, **kwargs): 
    predictions = []
    golds = []
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    with open(f"{model_augment_path}/{language}_results.txt", 'r') as predictions_f:
        num_blocks = 0
        while not predictions_f.readline().startswith("Generate"): # this skips the source
            # predictions_f.readline() # skip source
            gold_line = predictions_f.readline()
            example_num = int((gold_line.split('\t')[0])[2:])
            gold = ''.join(gold_line.split('\t')[1].strip().split(' '))
            if example_num < num_test_examples:
                golds.append(gold)

            hypothesis_line = predictions_f.readline()
            hypothesis = ''.join(hypothesis_line.split('\t')[2].strip().split(' '))
            if example_num < num_test_examples:
                predictions.append(hypothesis)
            predictions_f.readline() # skip Detokenized line
            predictions_f.readline() # skip per token line
            num_blocks += 1
            if num_blocks % 10 == 0:
                print(f"Extracted {num_blocks} hypotheses")
    predictions_and_golds = zip(predictions, golds)

    total = 0
    num_correct = 0
    for (prediction, gold) in predictions_and_golds:
        if prediction == gold:
            num_correct += 1
        total += 1
    assert total == num_test_examples
    with open(f"{model_augment_path}/final_results.txt", 'w') as accuracy_res_f:
        accuracy_res_f.write(f"For language {language}, we obtain an accuracy of {num_correct/total} when using augmentation strategy {augmentation_type}")

def extract_log_likelihoods(language, num_test_examples, **kwargs ):
    """Extracts log likelihoods only for augmented datapoints. 
    Assumes that the generation file has hypotheses that are numbered
    so that {S-[0...num_test_examples]} refer to the gold test examples, while
    everything after refers to the augmented examples.

    Args:
        language (str) 
        num_test_examples (int, optional): Number of gold test examples. 
    """
    aug_pool_size = kwargs['aug_pool_size']
    avg_log_likelihoods = []
    initial_model_path = get_model_augment_path(language, 'initial', **kwargs)
    with open(f"{initial_model_path}/{language}_results.txt", 'r') as predictions_f:
        while not predictions_f.readline().startswith("Generate"): # this skips the source
            predictions_f.readline() # skip target line
            hypothesis_line = predictions_f.readline()
            example_num = int(hypothesis_line.split('\t')[0][2:])
            confidence = float(hypothesis_line.split('\t')[1].strip())
            if example_num >= num_test_examples:
                avg_log_likelihoods.append((example_num, confidence))
            predictions_f.readline() # skip Detokenized line
            predictions_f.readline() # skip per token line
    with open(f"{initial_model_path}/{language}_log_likelihoods.pickle", "wb") as ll_handle: 
        pickle.dump(avg_log_likelihoods, ll_handle, protocol=pickle.HIGHEST_PROTOCOL)
    assert len(avg_log_likelihoods) == aug_pool_size , f"There were {len(avg_log_likelihoods)} log likelihoods collected but {aug_pool_size} were expected." 

def get_initial_model_path(language, **kwargs):
    init_kwargs = {k: kwargs[k] for k in INITIAL_MODEL_PARAMS}
    initial_augment_path = get_model_augment_path(language, 'initial', **init_kwargs)
    return initial_augment_path

def prep_preproc_fairseq_data_augment(language, augmentation_type, **kwargs):
    aug_pool_size = kwargs['aug_pool_size']
    train_frame, validation_frame, test_frame = load_gold_train_validation_test(language, kwargs['train_medium'])
    initial_generation_frame = get_initial_generation_frame(language, aug_pool_size) # contains gold test + original 10,000 test examples.
    num_gold_test_examples = get_number_test_examples(language)
    augment_example_lengths = get_augmentation_example_lengths(initial_generation_frame, num_gold_test_examples)
    subset_sampler = get_subset_selecter(language, augmentation_type, get_initial_model_path(language, **kwargs), initial_generation_frame, num_gold_test_examples, augment_example_lengths, **kwargs)
    # TODO: need to prefix with the number of points that are selected.
    subset_augmentation_frame = subset_sampler.get_best_points(kwargs['num_aug']) 

    train_augmented_frame = pd.concat([train_frame, subset_augmentation_frame])
    _write_split(language, augmentation_type, train_augmented_frame, "train", **kwargs)
    _write_split(language, augmentation_type, validation_frame, "valid", **kwargs)
    _write_split(language, augmentation_type, test_frame, "test", **kwargs)

def run_initial_pipeline(language: str, train_medium: bool, rand_seed: int, aug_pool_size: int):
    hparam_comb = {
        "train_medium": train_medium, 
        "rand_seed": rand_seed, 
        "aug_pool_size": aug_pool_size
    }
    prep_preproc_fairseq_data_initial(language, 'initial', **hparam_comb)
    run_fairseq_binarizer(language, 'initial', **hparam_comb)
    train_model(language, 'initial', **hparam_comb)
    generate(language, 'initial', **hparam_comb)
    report_accuracy(language, 'initial', get_number_test_examples(language), **hparam_comb)
    extract_log_likelihoods(language, get_number_test_examples(language), **hparam_comb)

def run_random_sampling_pipeline(language, num_aug, train_medium, rand_seed, aug_pool_size): 
    hparam_comb = {
        "num_aug": num_aug,
        "train_medium": train_medium, 
        "rand_seed": rand_seed, 
        "aug_pool_size": aug_pool_size
    }
    prep_preproc_fairseq_data_augment(language, 'random', **hparam_comb)
    run_fairseq_binarizer(language, 'random', **hparam_comb)
    train_model(language, 'random', **hparam_comb)
    generate(language, 'random', **hparam_comb)
    report_accuracy(language, 'random', get_number_test_examples(language), **hparam_comb)

def run_uncertainty_sampling_pipeline(language, num_aug, \
                                      use_high_loss, train_medium, \
                                      rand_seed, aug_pool_size): 
    hparam_comb = {
        "num_aug": num_aug,
        "use_high_loss": use_high_loss,
        "train_medium": train_medium,
        "rand_seed": rand_seed,
        "aug_pool_size": aug_pool_size 
    }
    prep_preproc_fairseq_data_augment(language, 'uncertainty_sample', **hparam_comb)
    run_fairseq_binarizer(language, 'uncertainty_sample', **hparam_comb)
    train_model(language, 'uncertainty_sample', **hparam_comb)
    generate(language, 'uncertainty_sample', **hparam_comb)
    report_accuracy(language, 'uncertainty_sample', get_number_test_examples(language), **hparam_comb)

def run_uat_pipeline(language: str, num_aug: int, train_medium: bool, \
                     use_empirical: bool, use_loss: bool, rand_seed: int, 
                     aug_pool_size: int):
    # TODO: put in the number of augmented examples into the map.
    hparam_comb = {
        "num_aug": num_aug,
        "use_empirical": use_empirical,
        "train_medium": train_medium,
        "rand_seed": rand_seed,
        "use_loss": use_loss,
        "aug_pool_size": aug_pool_size
    }
    algorithm = 'uat'
    prep_preproc_fairseq_data_augment(language, algorithm, **hparam_comb)
    run_fairseq_binarizer(language, algorithm, **hparam_comb)
    train_model(language, algorithm, **hparam_comb)
    generate(language, algorithm, **hparam_comb)
    report_accuracy(language, algorithm, get_number_test_examples(language), **hparam_comb)

def main(args):
    # atomic
    if args.prep_preproc_fairseq_data_initial:
        prep_preproc_fairseq_data_initial(args.language, args.augmentation_type)
    if args.prep_preproc_fairseq_data_augment:
        prep_preproc_fairseq_data_augment(args.language, args.augmentation_type, diverse_sample_k=args.diverse_sample_k)
    elif args.run_fairseq_binarizer:
        run_fairseq_binarizer(args.language, args.augmentation_type) # TODO: 
    elif args.train_model:
        train_model(args.language, args.augmentation_type)
    elif args.generate:
        generate(args.language, args.augmentation_type)
    elif args.report_accuracy:
        report_accuracy(args.language, args.augmentation_type, get_number_test_examples(args.language))
    elif args.extract_log_likelihoods:
        extract_log_likelihoods(args.language, get_number_test_examples(args.language))

    # pipelines
    elif args.run_initial_pipeline:
        run_initial_pipeline(args.language, args.train_medium, args.rand_seed, args.aug_pool_size)
    elif args.run_uncertainty_sampling_pipeline:
        run_uncertainty_sampling_pipeline(args.language, args.num_aug, \
            args.use_high_loss, args.train_medium, args.rand_seed, args.aug_pool_size)
    elif args.run_random_sampling_pipeline:
        run_random_sampling_pipeline(args.language, args.num_aug, \
            args.train_medium, args.rand_seed, args.aug_pool_size)
    elif args.run_uat_pipeline:
        run_uat_pipeline(args.language, args.num_aug, args.train_medium, \
            args.use_empirical, args.use_loss, args.rand_seed, args.aug_pool_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str)
    parser.add_argument("rand_seed", type=int)
    parser.add_argument("augmentation_type", type=str) # when starting out, just put "initial".
    parser.add_argument("num_aug", type=int) 
    parser.add_argument("aug_pool_size", type=int) 
    parser.add_argument("--prep_preproc_fairseq_data_initial", action='store_true')
    parser.add_argument("--prep_preproc_fairseq_data_augment", action='store_true')
    parser.add_argument("--probe_initial_representations", action='store_true')
    parser.add_argument("--train_model", action='store_true')
    parser.add_argument("--run_fairseq_binarizer", action='store_true')
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--report_accuracy", action='store_true')
    parser.add_argument("--extract_log_likelihoods", action='store_true')
    parser.add_argument("--diverse_sample_k", type=int, default=0)
    parser.add_argument("--use_empirical", action='store_true')

    parser.add_argument("--run_initial_pipeline", action='store_true')
    
    # Uncertainty sampling
    parser.add_argument("--use_high_loss", action='store_true')

    # for uat pipeline
    parser.add_argument("--use_loss", action='store_true') 

    # all options
    parser.add_argument("--train_medium", action='store_true')

    # Pipelines
    parser.add_argument("--run_uncertainty_sampling_pipeline", action='store_true')
    parser.add_argument("--run_random_sampling_pipeline", action='store_true')
    parser.add_argument("--run_uat_pipeline", action='store_true')
    main(parser.parse_args())