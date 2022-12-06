import numpy as np
import pickle
import pdb
import subprocess
import os 
import argparse
import pandas as pd
from collections import defaultdict

from packages.utils.constants import SIGM_DATA_PATH, SCRATCH_PATH, FAIRSEQ_SCRIPTS_PATH
from packages.augmentation.select_highest_loss import HighLossSampler
from packages.augmentation.subset_selecter_strategy import get_subset_selecter
from packages.fairseq_utils.dataloading_utils import get_initial_generation_frame
from packages.utils.util_functions import get_model_augment_path

def tokenize_row_src(row):
    tokens = list(row.src) + row.tag.split(";")
    return " ".join(tokens)

def tokenize_row_tgt(row):
    tokens = list(row.tgt)
    return " ".join(tokens)

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

def load_gold_train_validation_test(language):
    train_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-train-low", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    validation_frame= pd.read_csv(f"{SIGM_DATA_PATH}/{language}-dev", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    return train_frame, validation_frame, test_frame

# TODO: do these for all files
def prep_preproc_fairseq_data_initial(language, augmentation_type):
    train_frame, validation_frame, test_frame = load_gold_train_validation_test(language)

    test_frame = get_initial_generation_frame(language)

    # TODO: we need to write this to a separate scratch directory.
    _write_split(language, augmentation_type, train_frame, "train-low")
    _write_split(language, augmentation_type, validation_frame, "valid")
    _write_split(language, augmentation_type, test_frame, "test")


# TODO: have to change everything below to work with the new type of directories.
def run_fairseq_binarizer(language, augmentation_type, **kwargs):
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/preprocess.sh", model_augment_path, language])
    print(f"Obtained {result} result")

def train_model(language, augmentation_type, **kwargs):
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    algorithm_type = model_augment_path.split('/')[-1]
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", model_augment_path, language, algorithm_type])
    print(f"Obtained {result} result")

def generate(language, augmentation_type, **kwargs):
    model_augment_path = get_model_augment_path(language, augmentation_type, **kwargs)
    algorithm_type = model_augment_path.split('/')[-1]
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/generate.sh", model_augment_path, language, algorithm_type])
    print(f"Obtained {result} result")

def rename_initial_embeddings(language):
    model_augment_path = get_model_augment_path(language, 'initial')
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/mv_initial_embeddings.sh", model_augment_path, language])
    print(f"Obtained {result} result for renaming initial embeddings")

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

def extract_log_likelihoods(language, num_test_examples ):
    """Extracts log likelihoods only for augmented datapoints. 
    Assumes that the generation file has hypotheses that are numbered
    so that {S-[0...num_test_examples]} refer to the gold test examples, while
    everything after refers to the augmented examples.

    Args:
        language (str) 
        num_test_examples (int, optional): Number of gold test examples. 
    """
    avg_log_likelihoods = []
    with open(f"{SCRATCH_PATH}/{language}/initial_/{language}_results.txt", 'r') as predictions_f:
        while not predictions_f.readline().startswith("Generate"): # this skips the source
            predictions_f.readline() # skip target line
            hypothesis_line = predictions_f.readline()
            example_num = int(hypothesis_line.split('\t')[0][2:])
            confidence = float(hypothesis_line.split('\t')[1].strip())
            if example_num >= num_test_examples:
                avg_log_likelihoods.append((example_num, confidence))
            predictions_f.readline() # skip Detokenized line
            predictions_f.readline() # skip per token line
    with open(f"{SCRATCH_PATH}/{language}/initial_/{language}_log_likelihoods.pickle", "wb") as ll_handle: 
        pickle.dump(avg_log_likelihoods, ll_handle, protocol=pickle.HIGHEST_PROTOCOL)

    assert len(avg_log_likelihoods) == 10000

def probe_initial_representations(language, number_test_examples):
    path = f"{SCRATCH_PATH}/{language}/initial"

    token_id_to_embeds = defaultdict(list)
    with open(f"{path}/{language}_embeddings.pickle", "rb") as embeddings_pkl, \
         open(f"{path}/{language}_src_tokens.pickle", "rb") as token_ids_pkl, \
         open(f"{path}/{language}_ids.pickle", "rb") as item_ids_pkl, \
         open(f"{path}/{language}_src_dict.pickle", "rb") as src_dict_pkl:
        embeddings_dict = pickle.load(embeddings_pkl)
        item_id_dict = pickle.load(item_ids_pkl)
        token_ids_dict = pickle.load(token_ids_pkl)
        src_dict = pickle.load(src_dict_pkl)

        processed_items = set([])
        for seq_len in embeddings_dict.keys():
            embeds = np.array(embeddings_dict[seq_len])
            token_ids = np.array(token_ids_dict[seq_len])
            item_ids = np.array(item_id_dict[seq_len])
            for i in range(token_ids.shape[0]): # iterating over number of items
                if item_ids[i] < number_test_examples:
                    processed_items.add(item_ids[i])
                    for j in range(token_ids.shape[1]): # iterating over sequence length
                        token_id_to_embeds[token_ids[i, j]].append(embeds[j, i])
        # TODO concatenate all of the embeddings in each token.
        assert 0 in processed_items
        # for k, v in token_id_to_embeds:
        #     token_id_to_embeds[k] = np.vstack(v).to
    pdb.set_trace()
    with open(f"{path}/{language}_token_id_to_embeds.pickle", "wb") as handle:
        pickle.dump(token_id_to_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)

def prep_preproc_fairseq_data_augment(language, augmentation_type, **kwargs):
    train_frame, validation_frame, test_frame = load_gold_train_validation_test(language)
    initial_generation_frame = get_initial_generation_frame(language) # contains gold test + original 10,000 test examples.
    num_gold_test_examples = get_number_test_examples(language)
    subset_sampler = get_subset_selecter(language, augmentation_type, f"{SCRATCH_PATH}/{language}/initial_", initial_generation_frame, num_gold_test_examples, **kwargs)
    # TODO: need to prefix with the number of points that are selected.
    subset_augmentation_frame = subset_sampler.get_best_points(kwargs['num_aug']) 

    train_augmented_frame = pd.concat([train_frame, subset_augmentation_frame])
    _write_split(language, augmentation_type, train_augmented_frame, "train-low", **kwargs)
    _write_split(language, augmentation_type, validation_frame, "valid", **kwargs)
    _write_split(language, augmentation_type, test_frame, "test", **kwargs)

def run_initial_pipeline(language, num_aug):
    prep_preproc_fairseq_data_initial(language, 'initial')
    run_fairseq_binarizer(language, 'initial')
    train_model(language, 'initial')
    generate(language, 'initial')
    report_accuracy(language, 'initial', get_number_test_examples(language))
    extract_log_likelihoods(language, get_number_test_examples(language))
    rename_initial_embeddings(language)

def run_k_diverse_sampling_pipeline(language, num_aug): 
    # k_range = [8, 16, 32, 64, 128, 256, 512]
    # k_range = [8, 16, 32, 64, 128, 256, 512]
    k_range = [32]
    for k in k_range:
        prep_preproc_fairseq_data_augment(language, 'diversity_sample', k=k, num_aug=num_aug)
        run_fairseq_binarizer(language, 'diversity_sample', k=k)
        train_model(language, 'diversity_sample', k=k)
        generate(language, 'diversity_sample', k=k)
        report_accuracy(language, 'diversity_sample', get_number_test_examples(language), k=k)

def run_random_sampling_pipeline(language, num_aug): 
    prep_preproc_fairseq_data_augment(language, 'random', num_aug=num_aug)
    run_fairseq_binarizer(language, 'random', num_aug=num_aug)
    train_model(language, 'random', num_aug=num_aug)
    generate(language, 'random', num_aug=num_aug)
    report_accuracy(language, 'random', get_number_test_examples(language), num_aug=num_aug)

def run_uncertainty_sampling_pipeline(language, num_aug): 
    prep_preproc_fairseq_data_augment(language, 'uncertainty_sample', num_aug=num_aug)
    run_fairseq_binarizer(language, 'uncertainty_sample')
    train_model(language, 'uncertainty_sample')
    generate(language, 'uncertainty_sample')
    report_accuracy(language, 'uncertainty_sample', get_number_test_examples(language))

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
    elif args.probe_initial_representations:
        probe_initial_representations(args.language, get_number_test_examples(args.language))
    elif args.extract_log_likelihoods:
        extract_log_likelihoods(args.language, get_number_test_examples(args.language))

    # pipelines
    elif args.run_initial_pipeline:
        run_initial_pipeline(args.language, args.num_aug)
    elif args.run_k_diverse_sampling_pipeline:
        run_k_diverse_sampling_pipeline(args.language, args.num_aug)
    elif args.run_uncertainty_sampling_pipeline:
        run_uncertainty_sampling_pipeline(args.language, args.num_aug)
    elif args.run_random_sampling_pipeline:
        run_random_sampling_pipeline(args.language, args.num_aug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str)
    parser.add_argument("augmentation_type", type=str) # when starting out, just put "initial".
    parser.add_argument("num_aug", type=int) # when starting out, just put "initial".
    parser.add_argument("--prep_preproc_fairseq_data_initial", action='store_true')
    parser.add_argument("--prep_preproc_fairseq_data_augment", action='store_true')
    parser.add_argument("--probe_initial_representations", action='store_true')
    parser.add_argument("--train_model", action='store_true')
    parser.add_argument("--run_fairseq_binarizer", action='store_true')
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--report_accuracy", action='store_true')
    parser.add_argument("--extract_log_likelihoods", action='store_true')
    parser.add_argument("--diverse_sample_k", type=int, default=0)

    parser.add_argument("--run_initial_pipeline", action='store_true')
    parser.add_argument("--run_k_diverse_sampling_pipeline", action='store_true')
    parser.add_argument("--run_uncertainty_sampling_pipeline", action='store_true')
    parser.add_argument("--run_random_sampling_pipeline", action='store_true')
    main(parser.parse_args())