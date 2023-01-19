import argparse
import click
import os 
import subprocess
import pandas as pd
from itertools import product
import tqdm

from packages.utils.constants import SIGM_DATA_PATH, SCRATCH_PATH, LANGUAGES
from packages.utils.util_functions import load_sigm_file, get_model_augment_path, tokenize_row_tgt, tokenize_row_src, construct_cg_test_set

def write_cg_test_frame(src_path, tgt_path, cg_test_frame: pd.DataFrame):
    if len(cg_test_frame) > 1000: # arbitrary.
        cg_test_frame = cg_test_frame.sample(n=1000, replace=False, random_state=0)
    with open(src_path, 'w') as cg_frame_f:
        cg_test_frame[['src', 'tag']].apply(lambda row: cg_frame_f.write(f"{tokenize_row_src(row)}\n"), axis=1)
    with open(tgt_path, 'w') as cg_frame_f:
        cg_test_frame[['tgt']].apply(lambda row: cg_frame_f.write(f"{tokenize_row_tgt(row)}\n"), axis=1)

def evaluate_on_cg_test_set(language):
    # NOTE: i hope there's no characters in this expanded test set that are not in the vocab...
    # TODO: write result into the same folder.
    prefix = f"{SCRATCH_PATH}/{language}"
    for dirname in tqdm.tqdm(os.listdir(prefix)):
        model_name = f"{prefix}/{dirname}/{language}_{dirname}_model_checkpoints/checkpoint_best.pt"
        model_preproc_dir = f"{prefix}/{dirname}/{language}_fairseq_bin" 
        if not os.path.isdir(f"{prefix}/{dirname}"):
            print(f"Skipping because {dirname} is not a directory")
            continue

        if not os.path.exists(model_preproc_dir): # TODO: should be an easy fix. Just look for the directory that ends with checkpoints.
            print(f"The preprocessing directory for {dirname} does not exist; skipping for now")
            continue
        if not os.path.exists(model_name):
            for possible_model_dir in os.listdir(f"{prefix}/{dirname}"):
                if 'model_checkpoints' in possible_model_dir: 
                    model_name = f"{prefix}/{dirname}/{possible_model_dir}/checkpoint_best.pt"
                    break
        if 'train_medium=True' in dirname:
            stdin = f"{SCRATCH_PATH}/{language}/cg_test_frame_medium_excluded_2.txt"
            stdout = f"{SCRATCH_PATH}/{language}/{dirname}/cg_medium_excluded_2.txt"
        elif 'train_medium=False' in dirname:
            stdin = f"{SCRATCH_PATH}/{language}/cg_test_frame_low_excluded_2.txt"
            stdout = f"{SCRATCH_PATH}/{language}/{dirname}/cg_low_excluded_2.txt"
        
        if os.path.exists(stdout):
            print(f"Skipping because it appears that {stdout} has already been generated")
            continue

        stdout = open(stdout, 'w')
        # subprocess.run(["fairseq-interactive", "--path", model_name, model_preproc_dir, 
        #                 "--source-lang", "src", "--target-lang", "tgt", # probably for how to encode and decode...
        #                 "--tokenizer", "space", "--buffer-size", "100"],
        #                 stdin=stdin, stdout=stdout, check=True) 

        #the required batch size multiple assumes that the number of sentences 
        # to be generated is a multiple of 4
        subprocess.run(["fairseq-generate", model_preproc_dir, "--path", model_name, 
                        "--batch-size", "128", "--beam", "5", "--required-batch-size-multiple", "4"], 
                        stdout=stdout, check=True)
        print(f"Completed {dirname}")

def write_test_frames(seed):
    for language_quantity in tqdm.tqdm(product(LANGUAGES, [ 'medium'])): 
        language = language_quantity[0]
        quantity = language_quantity[1]
        test_frame = construct_cg_test_set(language, quantity)
        write_cg_test_frame(f"{SCRATCH_PATH}/{language}_seed={seed}/cg_test_frame_{quantity}_excluded_2.txt", \
                            f"{SCRATCH_PATH}/{language}_seed={seed}/cg_test_frame_{quantity}_excluded_2_tgt.txt", test_frame)

def main(args):
    seed = args.rand_seed
    write_test_frames(seed)
    for language in LANGUAGES:
        evaluate_on_cg_test_set(language)

parser = argparse.ArgumentParser()
parser.add_argument("rand_seed", type=int)
main(parser.parse_args())