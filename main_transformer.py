import numpy as np
import pickle
import pdb
import subprocess
import os 
import argparse
import pandas as pd
from collections import defaultdict

from sklearn.covariance import log_likelihood

from packages.utils.constants import SIGM_DATA_PATH, SCRATCH_PATH, FAIRSEQ_SCRIPTS_PATH

def tokenize_row_src(row):
    tokens = list(row.src) + row.tag.split(";")
    return " ".join(tokens)

def tokenize_row_tgt(row):
    tokens = list(row.tgt)
    return " ".join(tokens)

def get_initial_generation_frame(language): # concatenation of test file and augmentation file.
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    augmentation_frame = pd.read_csv(f"data/spreadsheets/augmentations/{language}-train-low-hall", header=None, names=["src", "tgt" ,"tag"], sep='\t')

    test_frame = pd.concat([test_frame, augmentation_frame]) # this is for the initial model, so we can get likelihoods and generations
    return test_frame


# TODO: do these for all files
def prep_preproc_fairseq_data_initial(language, augmentation_type):
    train_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-train-low", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    validation_frame= pd.read_csv(f"{SIGM_DATA_PATH}/{language}-dev", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    augmentation_frame = pd.read_csv(f"data/spreadsheets/augmentations/{language}-train-low-hall", header=None, names=["src", "tgt" ,"tag"], sep='\t')

    # TODO: replace with get_initial_generation_frame 
    test_frame = get_initial_generation_frame([test_frame, augmentation_frame]) # this is for the initial model, so we can get likelihoods and generations

    # TODO: we need to write this to a separate scratch directory.
    def _write_split(split_frame, split_name):
        if not os.path.exists(f"{SCRATCH_PATH}/{language}/{augmentation_type}"):
            os.makedirs(f"{SCRATCH_PATH}/{language}/{augmentation_type}")

        # write src
        with open(f"{SCRATCH_PATH}/{language}/{augmentation_type}/{language}-{split_name}.src", "w") as fseq_src_f:
            split_frame.apply(lambda row: fseq_src_f.write(f"{tokenize_row_src(row)}\n"), 
                            axis=1)  # rows
        # write tgt
        with open(f"{SCRATCH_PATH}/{language}/{augmentation_type}/{language}-{split_name}.tgt", "w") as fseq_tgt_f:
            split_frame.apply(lambda row: fseq_tgt_f.write(f"{tokenize_row_tgt(row)}\n"), 
                            axis=1)  # rows
    _write_split(train_frame, "train-low")
    _write_split(validation_frame, "valid")
    _write_split(test_frame, "test")

def run_fairseq_binarizer(language, augmentation_type):
    # TODO: need to append the language to scratch_path
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/preprocess.sh", f"{SCRATCH_PATH}/{language}/{augmentation_type}", language])
    print(f"Obtained {result} result")

def train_model(language, augmentation_type):
    # preproc_dir = f"{SCRATCH_PATH}/{language}_fairseq_bin"

    # save_dir = f"{SCRATCH_PATH}/{language}_model_checkpoints"
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/train_model.sh", f"{SCRATCH_PATH}/{language}/{augmentation_type}", language])
    print(f"Obtained {result} result")

def generate(language, augmentation_type):
    result = subprocess.run([f"{FAIRSEQ_SCRIPTS_PATH}/generate.sh", f"{SCRATCH_PATH}/{language}/{augmentation_type}", language])
    print(f"Obtained {result} result")

# TODO: this will no longer work exactly right, since we generate the augmentation predictions at the same time.
def report_accuracy_initial(language, num_test_examples=100): # TODO: do all the test files have exactly 100 cases?
    predictions = []
    golds = []
    with open(f"{SCRATCH_PATH}/{language}/initial/{language}_results.txt", 'r') as predictions_f:

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
        total +=1
    assert total == 100
    print(f"Accuracy of {num_correct/total}")

def extract_log_likelihoods(language, num_test_examples = 100):
    avg_log_likelihoods = []
    with open(f"{SCRATCH_PATH}/{language}/initial/{language}_results.txt", 'r') as predictions_f:
        while not predictions_f.readline().startswith("Generate"): # this skips the source
            predictions_f.readline() # skip target line
            hypothesis_line = predictions_f.readline()
            example_num = int(hypothesis_line.split('\t')[0][2:])
            confidence = float(hypothesis_line.split('\t')[1].strip())
            if example_num >= num_test_examples:
                avg_log_likelihoods.append((example_num, confidence))
            predictions_f.readline() # skip Detokenized line
            predictions_f.readline() # skip per token line
    with open(f"{SCRATCH_PATH}/{language}/initial/{language}_log_likelihoods.pickle", "wb") as ll_handle: 
        pickle.dump(avg_log_likelihoods, ll_handle, protocol=pickle.HIGHEST_PROTOCOL)

    assert len(avg_log_likelihoods) == 10000

def probe_initial_representations(language):
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
                if item_ids[i] < 100: # 100 is number of ground truth items
                    processed_items.add(item_ids[i])
                    for j in range(token_ids.shape[1]): # iterating over sequence length
                        token_id_to_embeds[token_ids[i, j]].append(embeds[j, i])
        # TODO concatenate all of the embeddings in each token.
        assert 0 in processed_items
        # for k, v in token_id_to_embeds:
        #     token_id_to_embeds[k] = np.vstack(v).to
    pdb.set_trace()
    with open(f"{path}/bengali_token_id_to_embeds.pickle", "wb") as handle:
        pickle.dump(token_id_to_embeds, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(args):
    if args.prep_preproc_fairseq_data:
        prep_preproc_fairseq_data_initial(args.language, args.augmentation_type)
    elif args.run_fairseq_binarizer:
        run_fairseq_binarizer(args.language, args.augmentation_type)
    elif args.train_model:
        train_model(args.language, args.augmentation_type)
    elif args.generate:
        generate(args.language, args.augmentation_type)
    elif args.report_accuracy:
        report_accuracy_initial(args.language)
    elif args.probe_initial_representations:
        probe_initial_representations(args.language)
    elif args.extract_log_likelihoods:
        extract_log_likelihoods(args.language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("language", type=str)
    parser.add_argument("augmentation_type", type=str) # when starting out, just put "initial".
    parser.add_argument("--prep_preproc_fairseq_data", action='store_true')
    parser.add_argument("--probe_initial_representations", action='store_true')
    parser.add_argument("--train_model", action='store_true')
    parser.add_argument("--run_fairseq_binarizer", action='store_true')
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--report_accuracy", action='store_true')
    parser.add_argument("--extract_log_likelihoods", action='store_true')
    main(parser.parse_args())