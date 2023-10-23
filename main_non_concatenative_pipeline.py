import loguru
import pickle
import shutil
import os
import ipdb
import polars as pl
import pandas as pd
from collections import OrderedDict
from flowmason import conduct, cacheable
import subprocess
from augment import retrieve_alignment_fails
from dotenv import dotenv_values
from typing import List, Tuple, Dict, Callable
from itertools import product
from functools import partial


from packages.utils.util_functions import construct_cg_test_set, get_model_augment_path, load_gold_train_validation_test, tokenize_row_src, tokenize_row_tgt
from packages.utils.constants import SIGM_DATA_PATH, FAIRSEQ_SCRIPTS_PATH, SCRATCH_PATH
from packages.augmentation.random_sampler import random_sample_augmented_data
from packages.augmentation.select_highest_loss import highest_uncertainty_sample_augmented_data

config = dotenv_values(".env")
cache_path = config['CACHE_PATH']
logger = loguru.logger

def step_load_arabic_test_dataset(step_name: str, version: str) -> pd.DataFrame:
    cg_test_set_arabic_frame = construct_cg_test_set("arabic", "low")
    return cg_test_set_arabic_frame

def step_load_medium_arabic_test_dataset(step_name: str, version: str) -> pd.DataFrame:
    cg_test_set_arabic_frame = construct_cg_test_set("arabic", "medium")
    return cg_test_set_arabic_frame

def step_load_non_concat_examples(step_name: str, version: str, cg_test_set_arabic_frame: pl.DataFrame):
    arabic_test_frame = pl.from_pandas(cg_test_set_arabic_frame)
    alignment_fails = retrieve_alignment_fails(arabic_test_frame['src'].apply(lambda x: x.strip()),
                                                arabic_test_frame['tgt'].apply(lambda x: x.strip()))
    arabic_test_frame = arabic_test_frame.with_columns(
        pl.lit(alignment_fails).alias("alignment_failed")
    )
    return arabic_test_frame

def step_generate_augmented_data(step_name: str, version: str) -> pd.DataFrame:
    language = "arabic"
    subprocess.run(["python", "augment.py", SIGM_DATA_PATH, language], check=True)
    augmentation_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-hall", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    return augmentation_frame

def _binarize_data(train_frame, validation_frame, test_dataframe, 
                   model_augment_path):
    if os.path.exists(f"{model_augment_path}"):
        shutil.rmtree(f"{model_augment_path}")
    if not os.path.exists(model_augment_path):
        os.makedirs(model_augment_path)
    language = "arabic"
    def _write_src_tgt_to_file(split_frame: pl.DataFrame, split_name: str):
        if type(split_frame) == pd.DataFrame:
            split_frame = pl.from_pandas(split_frame)
        with open(f"{model_augment_path}/{language}-{split_name}.src", "w") as fseq_src_f:
            split_frame.select(['src', 'tag']).map_rows(lambda row: fseq_src_f.write(f"{tokenize_row_src(row)}\n"))  # rows
        # write tgt
        with open(f"{model_augment_path}/{language}-{split_name}.tgt", "w") as fseq_tgt_f:
            split_frame.select('tgt').map_rows(lambda row: fseq_tgt_f.write(f"{tokenize_row_tgt(row)}\n"))
    _write_src_tgt_to_file(train_frame, "train")
    _write_src_tgt_to_file(validation_frame, "valid")
    _write_src_tgt_to_file(test_dataframe, "test")
    subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/preprocess.sh", model_augment_path, language], check=True)
    assert len(os.listdir(f"{model_augment_path}/{language}_fairseq_bin")) == 4 * 3 + 3 # 4 files for each of the 3 splits + 3 files for the dictionary

def step_binarize_initial_training_and_eval_data(step_name: str, version: str,
                             cg_test_frame: pl.DataFrame, augmentation_frame: pd.DataFrame):
    language = 'arabic'
    train_frame, validation_frame, _ = load_gold_train_validation_test(language, train_medium=False)
    augmentation_frame = pl.from_pandas(augmentation_frame)
    assert len(augmentation_frame) == 10000
    model_augment_path = get_model_augment_path(language, "initial", rand_seed=0, aug_pool_size=len(augmentation_frame))
    
    # remove the f"{model_augment_path}/{language}_fairseq_bin" directory

    eval_frame = pl.concat([cg_test_frame.select(['src', 'tgt', 'tag']),\
                                       augmentation_frame])
    _binarize_data(train_frame, validation_frame, eval_frame, model_augment_path)
    return f"{model_augment_path}/{language}_fairseq_bin"

def step_binarize_medium_training_and_eval_data(step_name: str, version: str,
                                cg_test_frame: pl.DataFrame, augmentation_frame: pd.DataFrame):
    language = 'arabic'
    train_frame, validation_frame, _ = load_gold_train_validation_test(language, train_medium=True)
    augmentation_frame = pl.from_pandas(augmentation_frame)
    assert len(augmentation_frame) == 10000
    model_augment_path = get_model_augment_path(language, "initial_medium", rand_seed=0, aug_pool_size=len(augmentation_frame))
    eval_frame = pl.concat([cg_test_frame.select(['src', 'tgt', 'tag']),\
                                        augmentation_frame])
    _binarize_data(train_frame, validation_frame, eval_frame, model_augment_path)
    return f"{model_augment_path}/{language}_fairseq_bin"

def step_train_initial_model(step_name: str, version: str, augmentation_frame: pd.DataFrame):
    language = 'arabic'
    model_augment_path = get_model_augment_path(language, "initial", rand_seed=0, aug_pool_size=len(augmentation_frame))
    algorithm_type = 'initial'
    rand_seed = "0"
    subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", model_augment_path, language, algorithm_type, rand_seed], check=True)
    return f"{model_augment_path}/{algorithm_type}_aug_pool_size={len(augmentation_frame)}/{language}_{algorithm_type}_model_checkpoints/checkpoint_best.pt"

def step_train_initial_medium_model(step_name: str, version: str, augmentation_frame: pd.DataFrame):
    language = 'arabic'
    model_augment_path = get_model_augment_path(language, "initial_medium", rand_seed=0, aug_pool_size=len(augmentation_frame))
    algorithm_type = 'initial_medium'
    rand_seed = "0"
    subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", model_augment_path, language, algorithm_type, rand_seed], check=True)
    return f"{model_augment_path}/{algorithm_type}_aug_pool_size={len(augmentation_frame)}/{language}_{algorithm_type}_model_checkpoints/checkpoint_best.pt"

# TODO: need to return the example number as well, so we know how to align the predictions with the source datapoint
def evaluate_generations_from_model(generation_file_path, num_test_examples: int):
    language = 'arabic'
    golds = []
    predictions = []
    example_nums = []
    with open(f"{generation_file_path}", 'r') as predictions_f:
        num_blocks = 0
        while not predictions_f.readline().startswith("Generate"): # this skips the source
            # predictions_f.readline() # skip source
            gold_line = predictions_f.readline()
            example_num = int((gold_line.split('\t')[0])[2:])
            gold = ''.join(gold_line.split('\t')[1].strip().split(' '))
            if example_num < num_test_examples:
                golds.append(gold)
                example_nums.append(example_num)

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
    assert total == num_test_examples, f"Expected {num_test_examples} examples but only {total} were found."
    return num_correct, total, list(zip(predictions, example_nums))

def step_generate_initial_model(step_name: str, version: str, cg_test_frame: pl.DataFrame,
                                augmentation_frame: pd.DataFrame):
    language = 'arabic'
    augmentation_type = 'initial_medium'
    model_augment_path = get_model_augment_path(language, augmentation_type, rand_seed=0, aug_pool_size=len(augmentation_frame))
    subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/generate.sh", model_augment_path, language, augmentation_type], check=True)

    num_test_examples = len(cg_test_frame)
    num_correct, total, _ = evaluate_generations_from_model(f"{model_augment_path}/{language}_results.txt", num_test_examples)
    with open(f"{model_augment_path}/final_results.txt", 'w') as accuracy_res_f:
        accuracy_res_f.write(f"For language {language}, we obtain an accuracy of {num_correct/total} when using augmentation strategy {augmentation_type}")
    return f"{model_augment_path}/{language}_results.txt"

def extract_log_likelihoods_aug_pool(cg_test_frame: pl.DataFrame, augmentation_type: str):
    aug_pool_size = 10000
    language = 'arabic'
    avg_log_likelihoods = []
    initial_model_path = get_model_augment_path(language, augmentation_type, rand_seed=0, aug_pool_size=aug_pool_size)
    num_test_examples = len(cg_test_frame)
    with open(f"{initial_model_path}/{language}_results.txt", 'r') as predictions_f:
        source_line = predictions_f.readline()
        found_zero_num = False
        while not source_line.startswith("Generate"): # this skips the source
            predictions_f.readline() # skip target line
            hypothesis_line = predictions_f.readline()
            example_num = int(hypothesis_line.split('\t')[0][2:])
            confidence = float(hypothesis_line.split('\t')[1].strip())
            if example_num >= num_test_examples:
                avg_log_likelihoods.append((example_num, confidence, source_line)) # NOTE: the source line is just for validation. Might break other functions
            if example_num == 0:
                found_zero_num = True
            predictions_f.readline() # skip Detokenized line
            predictions_f.readline() # skip per token line
            source_line = predictions_f.readline()
    assert found_zero_num, "The example numbers might not be zero generated"
    assert len(avg_log_likelihoods) == aug_pool_size , f"There were {len(avg_log_likelihoods)} log likelihoods collected but {aug_pool_size} were expected." 
    return avg_log_likelihoods
    
def step_extract_log_likelihoods_aug_pool(step_name: str, version: str,
                                          cg_test_frame: pl.DataFrame ):
    augmentation_type = 'initial'
    return extract_log_likelihoods_aug_pool(cg_test_frame, augmentation_type)

def step_extract_log_likelihoods_aug_pool_medium(step_name: str, version: str, cg_test_frame: pl.DataFrame):
    augmentation_type = 'initial_medium'
    return extract_log_likelihoods_aug_pool(cg_test_frame, augmentation_type)

def _retrieve_augmentation_frame_with_nll(augmentation_frame: pd.DataFrame, avg_log_likelihoods: List[Tuple[int, float, str]]):
    augmentation_frame = pl.from_pandas(augmentation_frame).with_columns(
        pl.lit(list(range(len(augmentation_frame)))).alias("augmentation_index")
    )
    nll_frame = pl.DataFrame({
        "nll": [x[1] for x in avg_log_likelihoods],
        "augmentation_index": [x[0] for x in avg_log_likelihoods]
    })
    augmentation_frame = augmentation_frame.join(nll_frame, on="augmentation_index")
    return augmentation_frame

def train_augmented_model(train_model_fn: Callable[[str, str, int, int], List[Tuple[str, str]]], cg_test_frame: pl.DataFrame):
    subset_sizes = [128, 512]
    seeds = [0, 1, 2]
    strategies = ['random']
    cg_test_with_results_frame = cg_test_frame
    cg_test_with_results_frame = cg_test_with_results_frame.with_columns(
        pl.lit(list(range(len(cg_test_with_results_frame)))).alias('datapoint_index')
    )
    for subset_size, seed, strategy in product(subset_sizes, seeds, strategies):
        predictions_example_nums = train_model_fn(version="005", strategy = strategy, 
                                                  seed = seed, subset_size=subset_size)
        prediction_frame = pl.DataFrame({
            f"prediction_ss={subset_size}_seed={seed}_strategy={strategy}": [prediction_example_num[0] for prediction_example_num in predictions_example_nums], 
            "datapoint_index": [prediction_example_num[1] for prediction_example_num in predictions_example_nums]
        })
        cg_test_with_results_frame = cg_test_with_results_frame.join(prediction_frame, on="datapoint_index")
    return cg_test_with_results_frame

# NOTE: the test frames will be different. 
def step_evaluate_initial_predictions(step_name: str, version: str, cg_test_frame_med: pl.DataFrame,
                                 cg_test_frame_low: pl.DataFrame):
    seeds = [0]
    language = 'arabic'
    frames = []
    cg_test_frame_low = cg_test_frame_low.with_columns(
        pl.lit(list(range(len(cg_test_frame_low)))).alias('datapoint_index')
    )
    cg_test_frame_med = cg_test_frame_med.with_columns(
        pl.lit(list(range(len(cg_test_frame_med)))).alias('datapoint_index')
    )
    for seed in seeds:
        model_augment_path = get_model_augment_path("arabic", 'initial', rand_seed=seed, aug_pool_size=10000)
        # evaluate_generations_from_model(f"{model_augment_path}/{language}_results.txt", len(cg_test_frame))
        num_correct, total, predictions_example_nums = evaluate_generations_from_model(f"{model_augment_path}/{language}_results.txt", len(cg_test_frame_low))
        initial_prediction_frame = pl.DataFrame({
            f"prediction_seed={seed}_strategy=initial": [prediction_example_num[0] for prediction_example_num in predictions_example_nums], 
            "datapoint_index": [prediction_example_num[1] for prediction_example_num in predictions_example_nums]
        })
        #print the accuracy
        logger.info(f"Obtained an accuracy of {num_correct/total} for initial with {len(cg_test_frame_low)} examples")

        model_augment_path = get_model_augment_path("arabic", 'initial_medium', rand_seed=seed, aug_pool_size=10000)
        num_correct, total, predictions_example_nums = evaluate_generations_from_model(f"{model_augment_path}/{language}_results.txt", len(cg_test_frame_med))
        initial_medium_prediction_frame = pl.DataFrame({
            f"prediction_seed={seed}_strategy=initial_medium": [prediction_example_num[0] for prediction_example_num in predictions_example_nums], 
            "datapoint_index": [prediction_example_num[1] for prediction_example_num in predictions_example_nums]
        })
        logger.info(f"Obtained an accuracy of {num_correct/total} for initial with {len(cg_test_frame_low)} examples")
        # join with the test frame
        cg_test_frame_low = cg_test_frame_low.join(initial_prediction_frame, on="datapoint_index")
        cg_test_frame_med = cg_test_frame_med.join(initial_medium_prediction_frame, on="datapoint_index")

    return cg_test_frame_low, cg_test_frame_med

def step_train_augmented_model_medium_setting(step_name: str, version: str, augmentation_frame: pd.DataFrame, 
                                              cg_test_frame: pl.DataFrame, avg_log_likelihoods: List[Tuple[int, float, str]]):

    augmentation_frame = _retrieve_augmentation_frame_with_nll(augmentation_frame, avg_log_likelihoods)
    language = 'arabic'
    aug_pool_size = 10000
    @cacheable(cache_dir=cache_path)
    def cacheable_train_augmented_model(cacheable_name: str, version: str,
                                        strategy: str, seed: int, subset_size: int) -> List[Tuple[str, str]]:
        if strategy == 'random':        
            augmentation_subset_frame = random_sample_augmented_data(augmentation_frame, subset_size)
        elif strategy == 'uncertainty':
            augmentation_subset_frame = highest_uncertainty_sample_augmented_data(augmentation_frame, subset_size)
        
        ## binarize
        train_frame, validation_frame, _ = load_gold_train_validation_test(language, train_medium=True)
        train_frame = pl.from_pandas(train_frame) if type(train_frame) == pd.DataFrame else train_frame
        augmentation_subset_frame = pl.from_pandas(augmentation_subset_frame) if type(augmentation_subset_frame) == pd.DataFrame else augmentation_subset_frame
        model_augment_path = get_model_augment_path(language, strategy, rand_seed=seed, num_aug=subset_size, aug_pool_size=aug_pool_size)
        train_frame = pl.concat([train_frame, augmentation_subset_frame.select(['src', 'tgt', 'tag'])])
        _binarize_data(train_frame, validation_frame, cg_test_frame, model_augment_path)

        ## train    
        algorithm_type = strategy
        subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", model_augment_path, language, algorithm_type, str(seed)], check=True)

        ## generate
        subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/generate.sh", model_augment_path, language, algorithm_type], check=True)
        num_correct, total, predictions_example_nums = evaluate_generations_from_model(f"{model_augment_path}/{language}_results.txt", len(cg_test_frame))
        logger.info(f"Obtained an accuracy of {num_correct/total} for {strategy} with {subset_size} examples")
        return predictions_example_nums
    return train_augmented_model(partial(
            cacheable_train_augmented_model, 
            cacheable_name="cacheable_train_augmented_model_medium"
        ), 
    cg_test_frame)

def step_train_augmented_model(step_name: str, version: str, 
                               augmentation_frame: pd.DataFrame, cg_test_frame: pl.DataFrame, 
                               avg_log_likelihoods: List[Tuple[int, float, str]]): 
    # attach avg_log_likelihoods to the augmentation frame
    aug_pool_size = 10000
    language = 'arabic'

    augmentation_frame = _retrieve_augmentation_frame_with_nll(augmentation_frame, avg_log_likelihoods)

    @cacheable(cache_dir=cache_path)
    def cacheable_train_augmented_model(cacheable_name: str, version: str,
                                        strategy: str, seed: int, subset_size: int) -> List[Tuple[str, str]]:
        if strategy == 'random':        
            augmentation_subset_frame = random_sample_augmented_data(augmentation_frame, subset_size)
        elif strategy == 'uncertainty':
            augmentation_subset_frame = highest_uncertainty_sample_augmented_data(augmentation_frame, subset_size)
        
        ## binarize
        train_frame, validation_frame, _ = load_gold_train_validation_test(language, train_medium=False)
        train_frame = pl.from_pandas(train_frame) if isinstance(train_frame, pd.DataFrame) else train_frame
        augmentation_subset_frame = pl.from_pandas(augmentation_subset_frame) if type(augmentation_subset_frame) == pd.DataFrame else augmentation_subset_frame
        model_augment_path = get_model_augment_path(language, strategy, rand_seed=seed, num_aug=subset_size, aug_pool_size=aug_pool_size)
        train_frame = pl.concat([train_frame, augmentation_subset_frame.select(['src', 'tgt', 'tag'])])
        _binarize_data(train_frame, validation_frame, cg_test_frame, model_augment_path)

        ## train    
        algorithm_type = strategy
        subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", model_augment_path, language, algorithm_type, str(seed)], check=True)

        ## generate
        subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/generate.sh", model_augment_path, language, algorithm_type], check=True)
        num_correct, total, predictions_example_nums = evaluate_generations_from_model(f"{model_augment_path}/{language}_results.txt", len(cg_test_frame))
        logger.info(f"Obtained an accuracy of {num_correct/total} for {strategy} with {subset_size} examples")
        return predictions_example_nums
    return train_augmented_model(partial(cacheable_train_augmented_model,
        cacheable_name="cacheable_train_augmented_model"),
    cg_test_frame)

def step_combine_low_and_med_predictions(step_name: str, version: str, low_pred_frame: pl.DataFrame,
                                         med_pred_frame: pl.DataFrame) -> pl.DataFrame:
    low_pred_frame = low_pred_frame.with_columns(
        pl.lit("low").alias("init_data_cond")
    )
    med_pred_frame = med_pred_frame.with_columns(
        pl.lit("medium").alias("init_data_cond")
    )
    return pl.concat([low_pred_frame, med_pred_frame])

def step_compute_accuracy_on_unaligned_datapoints(step_name: str,
                                                  version: str,
                                                  prediction_frame: pl.DataFrame, 
                                                  initial_prediction_frames: Tuple[pl.DataFrame, pl.DataFrame]):
    seeds = [0, 1, 2]
    subset_sizes = [128, 512]
    result_frames = []
    for seed, subset_size in product(seeds, subset_sizes):
        print(f"Computing accuracy for seed {seed} and subset size {subset_size}")
        result_frames.append(
            prediction_frame.with_columns([
                (pl.col('tgt') == pl.col(f'prediction_ss={subset_size}_seed={seed}_strategy=random')).alias('predictions_correct'),
                pl.lit(seed).alias('seed'),
                pl.lit(subset_size).alias('subset_size'),
            ]).group_by('alignment_failed', 'init_data_cond').agg(
                pl.col('predictions_correct').sum()/pl.col('predictions_correct').count(),
                pl.col('seed').first(),
                pl.col('subset_size').first(),
            )
        )
    result_frame = pl.concat(result_frames)
    print(result_frame)

    # compute accuracy for initial predictions, grouping by whether the alignment failed
    initial_prediction_frame_low, initial_prediction_frame_med = initial_prediction_frames
    ipdb.set_trace()
    initial_prediction_frame_low_results = initial_prediction_frame_low.with_columns([
        (pl.col('tgt') == pl.col('prediction_seed=0_strategy=initial')).alias('predictions_correct'),
        pl.lit(0).alias('seed'),
    ]).group_by('alignment_failed').agg(
        pl.col('predictions_correct').sum()/pl.col('predictions_correct').count(),
        pl.col('seed').first()
    )

    initial_prediction_frame_med_result = initial_prediction_frame_med.with_columns([
        (pl.col('tgt') == pl.col('prediction_seed=0_strategy=initial_medium')).alias('predictions_correct'),
        pl.lit(0).alias('seed'),
    ]).group_by('alignment_failed').agg(
        pl.col('predictions_correct').sum()/pl.col('predictions_correct').count(),
        pl.col('seed').first()
    )
    ipdb.set_trace()

if __name__ == "__main__":
    steps = OrderedDict()
    steps['load_arabic_test_dataset'] = (step_load_arabic_test_dataset, {
        "step_name": "step_load_arabic_test_dataset",
        "version": "001",
    })
    steps['load_medium_excl_arabic_test_dataset'] = (step_load_medium_arabic_test_dataset, {
        "step_name": "step_load_medium_arabic_test_dataset",
        "version": "001",
    })
    steps['load_non_concat_examples'] = (step_load_non_concat_examples, {
        "step_name": "step_load_non_concat_examples",
        "version": "003",
        "cg_test_set_arabic_frame": "load_arabic_test_dataset",
    })
    steps['load_non_concat_examples_medium'] = (step_load_non_concat_examples, {
        "step_name": "step_load_non_concat_examples_medium",
        "version": "001",
        "cg_test_set_arabic_frame": "load_medium_excl_arabic_test_dataset",
    })
    steps["generate_augmented_data"] = (step_generate_augmented_data, { # this is fixed for whatevre choice of initial training data
        "step_name": "step_generate_augmented_data",
        "version": "001",
    })
    steps["binarize_initial_data"] = (step_binarize_initial_training_and_eval_data, { 
        "version": "001",
        "cg_test_frame": "load_non_concat_examples",
        "augmentation_frame": "generate_augmented_data"
    })
    steps["binarize_initial_medium_data"] = (step_binarize_medium_training_and_eval_data, {
        "version": "001",
        "cg_test_frame": "load_non_concat_examples_medium",
        "augmentation_frame": "generate_augmented_data"
    })
    steps['train_initial_model'] = (step_train_initial_model, {
        "step_name": "step_train_initial_model",
        "version": "001",
        "augmentation_frame": "generate_augmented_data"
    })
    steps['train_initial_medium_model'] = (step_train_initial_medium_model, {
        "step_name": "step_train_initial_medium_model",
        "version": "001",
        "augmentation_frame": "generate_augmented_data"
    })

    # add step for generating from initial model
    steps['generate_initial_model'] = (step_generate_initial_model, {
        "step_name": "step_generate_initial_model",
        "version": "001",
        "cg_test_frame": "load_non_concat_examples",
        "augmentation_frame": "generate_augmented_data"
    })
    steps['generate_initial_model_medium'] = (step_generate_initial_model, {
        "step_name": "step_generate_initial_model_medium",
        "version": "001",
        "cg_test_frame": "load_non_concat_examples_medium",
        "augmentation_frame": "generate_augmented_data"
    })
    # add step for evaluating initial models
    steps['evaluate_initial_predictions'] = (step_evaluate_initial_predictions, {
        "step_name": "step_evaluate_initial_predictions",
        "version": "005",
        "cg_test_frame_med": "load_non_concat_examples_medium",
        "cg_test_frame_low": "load_non_concat_examples"
    })
    # add step for extracting log likelihoods from initial model
    steps['extract_log_likelihoods_aug_pool'] = (step_extract_log_likelihoods_aug_pool, {
        "step_name": "step_extract_log_likelihoods_aug_pool",
        "version": "001",
        "generation_results_path": "generate_initial_model",
        "cg_test_frame": "load_non_concat_examples",
        "augmentation_frame": "generate_augmented_data"
    })
    steps['extract_log_likelihoods_aug_pool_medium'] = (step_extract_log_likelihoods_aug_pool_medium, {
        "step_name": "step_extract_log_likelihoods_aug_pool_medium",
        "version": "001",
        "cg_test_frame": "load_non_concat_examples_medium",
    })
    # add step for training with augmented data
    steps['train_augmented_model'] = (step_train_augmented_model, {
        "step_name": "step_train_augmented_model",
        "version": "006",
        "cg_test_frame": "load_non_concat_examples",
        "augmentation_frame": "generate_augmented_data",
        "avg_log_likelihoods": "extract_log_likelihoods_aug_pool"
    })
    steps['train_augmented_model_medium'] = (step_train_augmented_model_medium_setting, {
        "step_name": "step_train_augmented_model_medium",
        "version": "001",
        "cg_test_frame": "load_non_concat_examples_medium",
        "augmentation_frame": "generate_augmented_data",
        "avg_log_likelihoods": "extract_log_likelihoods_aug_pool_medium"
    })
    steps['combine_low_and_med_predictions'] = (step_combine_low_and_med_predictions, {
        "step_name": "step_combine_low_and_med_predictions",
        "version": "001",
        "low_pred_frame": "train_augmented_model",
        "med_pred_frame": "train_augmented_model_medium"
    })

    steps['compute_accuracy_on_unaligned_datapoints'] = (step_compute_accuracy_on_unaligned_datapoints, {
        "step_name": "step_compute_accuracy_unaligned_data", 
        "version": "001", 
        "prediction_frame": "combine_low_and_med_predictions",
        "initial_prediction_frames": "evaluate_initial_predictions"
    })
    conduct(cache_path, steps, "nonconcatenative_experiments")