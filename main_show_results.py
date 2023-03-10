import numpy as np
from typing import TextIO
import re
import sys
import pdb
import argparse
import os
from tqdm import tqdm
from datetime import datetime

import pandas as pd
from packages.utils.constants import SCRATCH_PATH, LANGUAGES
from packages.utils.util_functions import get_top_path



def show_results(language, seed, aug_pool_size):
    prefix_dir = get_top_path(language, seed, aug_pool_size)
    dirs = os.listdir(prefix_dir)
    methods = []
    results = []
    for cur_dir in dirs:
        if not os.path.isdir(f"{prefix_dir}/{cur_dir}"):
            continue
        result_fname = f"{prefix_dir}/{cur_dir}/final_results.txt"
        try:
            with open(result_fname, 'r' , encoding='ascii') as result_f:
                result_line = result_f.readline().strip()
                result = re.search(r"0\.\d+", result_line)[0]
                print(f"Method {cur_dir} obtained result {result}")
                methods.append(cur_dir)
                results.append(result)
        except FileNotFoundError:
            print(f"Result file does not exist for {result_fname}; did the training crash?")
    frame = pd.DataFrame(
        {
            "method": methods, 
            "result": results 
        }
    )
    frame.to_csv(f"{prefix_dir}/{language}_results.csv")
    # assert len(frame) >= 52, language
    return frame 

def obtain_predictions_interactive(predictions_f: TextIO):
    cur_line = predictions_f.readline()
    predictions = []
    tags = []
    while not cur_line == '':
        tag_line = cur_line.split('\t')[1]
        # the tag is the sequence of tokens (separated by ' ') starting from either 'V' or 'N' in the tag line.
        # it is a sequence of tokens looking like 'V IND PRS 2 SG' or 'V IND PST 3 PL IPFV'.
        # there will be oome number of non-tag tokens before the tag, e.g., d e t e r m i n a r V IND PRS 2 SG.
        # we want to extract the tag, which is the last sequence of tokens in the line.
        tokens = tag_line.split(' ')
        cur_token = 0
        while cur_token < len(tokens) and tokens[cur_token] not in ['V', 'N']:
            cur_token += 1
        tag = ' '.join(tokens[cur_token:])
        tags.append(tag)
        predictions_f.readline() # skip generation time
        prediction_line = predictions_f.readline().strip()
        prediction = prediction_line.split('\t')[2]
        prediction = ''.join(prediction.split(' '))
        predictions.append(prediction)
        predictions_f.readline() # skip second prediction line
        predictions_f.readline() # skip per-token likelihood
        cur_line = predictions_f.readline() # start of new block
    return predictions, tags

def show_results_compositional(language: str, seed: int, aug_pool_size: int):
    prefix_dir = get_top_path(language, seed, aug_pool_size)
    dirs = os.listdir(prefix_dir)
    methods = []
    results = []
    data_quantity = []
    num_datapoints = []
    macro_accs = []
    macro_stds = []

    failed_dirs = []
    join_tokens = lambda s: ''.join(s.split(' '))
    for cur_dir in tqdm(dirs):
        if not os.path.isdir(f"{prefix_dir}/{cur_dir}"):
            continue
        if 'train_medium=True' in cur_dir:
            curr_quantity = 'medium'
            result_file = 'cg_medium_excluded_2.txt'
            with open(f'{prefix_dir}/cg_test_frame_medium_excluded_2_tgt.txt', 'r') as targets_f:
                targets = [join_tokens(line.strip()) for line in targets_f]
        elif 'train_medium=False' in cur_dir:
            curr_quantity = 'low'
            result_file = 'cg_low_excluded_2.txt'
            with open(f'{prefix_dir}/cg_test_frame_low_excluded_2_tgt.txt', 'r') as targets_f:
                targets = [join_tokens(line.strip()) for line in targets_f]
        else:
            raise Exception("wtf b")

        try: 
            with open(f"{prefix_dir}/{cur_dir}/{result_file}", 'r') as predictions_f:
                predictions, tags = obtain_predictions_interactive(predictions_f)
                try: 
                    assert len(predictions) == len(targets), f"len predictions is {len(predictions)} but the ground truth is of length {len(targets)}. Failed on {cur_dir}"
                except AssertionError as e:
                    failed_dirs.append(cur_dir)
                    continue
        except FileNotFoundError:
            print(f"File not found for: {cur_dir}")
            continue

        correct_series = pd.Series(predictions) == pd.Series(targets)
        data_quantity.append(curr_quantity)
        result = sum(correct_series)/len(correct_series)
        results.append(result)

        methods.append(cur_dir)
        num_datapoints.append(len(predictions))

        # compute the macro averaged accuracy over the tags.
        # create a dataframe with the predictions, targets, and tags.
        df = pd.DataFrame({
            'prediction': predictions,
            'target': targets,
            'tag': tags
        })
        # group by tag and compute the accuracy for each tag.
        tag_grouped = df.groupby('tag')
        tag_accs = tag_grouped.apply(lambda x: sum(x['prediction'] == x['target'])/len(x))
        # compute the macro average of the tag accuracies.
        macro_avg = sum(tag_accs)/len(tag_accs)
        macro_accs.append(macro_avg)
        # also compute standard deviation of the tag accuracies.
        macro_std = np.std(tag_accs)
        macro_stds.append(macro_std)

        
    print(f"All failed directories: {failed_dirs}")
    frame = pd.DataFrame({
        "method": methods,
        "result": results,
        "num_eval_datapoints": num_datapoints,
        "macro_avg_acc": macro_accs,
        "macro_std": macro_stds,
    })
    return frame

def main(args):
    seed = args.rand_seed
    aug_pool_size = args.aug_pool_size

    if args.show_results:
        all_frames = []
        for language in LANGUAGES:
            frame = show_results(language, seed, aug_pool_size) 
            frame['language'] = [language] * len(frame)
            pd.set_option('display.max_colwidth', None)
            all_frames.append(frame)
        all_results_frame = pd.concat(all_frames)
        cur_date = datetime.today().date().isoformat()
        all_results_frame.to_csv(f"{SCRATCH_PATH}/all_languages_results_{cur_date}.csv")
    elif args.show_results_compositional:
        all_frames = []
        for language in LANGUAGES:
            frame = show_results_compositional(language, seed, aug_pool_size)
            frame['language'] = [language] * len(frame)
            all_frames.append(frame)
        all_results_frame = pd.concat(all_frames)
        cur_date = datetime.today().date().isoformat()
        all_results_frame.to_csv(f"{SCRATCH_PATH}/all_languages_results_{cur_date}_seed={seed}_aug_pool_size={aug_pool_size}_compositional.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rand_seed", type=int)
    parser.add_argument("aug_pool_size", type=int)
    parser.add_argument("--show_results", action='store_true')
    parser.add_argument("--show_results_compositional", action='store_true')
    main(parser.parse_args())
