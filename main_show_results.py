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
from packages.utils.util_functions import load_sigm_file, construct_cg_test_set

def show_results(language, seed):
    prefix_dir = f"{SCRATCH_PATH}/{language}_seed={seed}"
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
    while not cur_line.startswith('Generate'):
        predictions_f.readline() # skip generation time
        prediction_line = predictions_f.readline().strip()
        prediction = prediction_line.split('\t')[2]
        prediction = ''.join(prediction.split(' '))
        predictions.append(prediction)
        predictions_f.readline() # skip second prediction line
        predictions_f.readline() # skip per-token likelihood
        cur_line = predictions_f.readline() # start of new block
    return predictions

def show_results_compositional(language: str, seed: int):
    prefix_dir = f"{SCRATCH_PATH}/{language}_seed={seed}"
    dirs = os.listdir(prefix_dir)
    methods = []
    results = []
    data_quantity = []
    num_datapoints = []

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
                predictions = obtain_predictions_interactive(predictions_f)
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
    print(f"All failed directories: {failed_dirs}")
    frame = pd.DataFrame({
        "method": methods,
        "result": results,
        "num_eval_datapoints": num_datapoints
    })
    return frame


def main(args):
    seed = args.rand_seed

    if args.show_results:
        all_frames = []
        for language in LANGUAGES:
            frame = show_results(language, seed)
            frame['language'] = [language] * len(frame)
            pd.set_option('display.max_colwidth', None)
            all_frames.append(frame)
        all_results_frame = pd.concat(all_frames)
        cur_date = datetime.today().date().isoformat()
        all_results_frame.to_csv(f"{SCRATCH_PATH}/all_languages_results_{cur_date}.csv")
    elif args.show_results_compositional:
        all_frames = []
        for language in LANGUAGES:
            frame = show_results_compositional(language, seed)
            frame['language'] = [language] * len(frame)
            all_frames.append(frame)
        all_results_frame = pd.concat(all_frames)
        cur_date = datetime.today().date().isoformat()
        all_results_frame.to_csv(f"{SCRATCH_PATH}/all_languages_results_{cur_date}_compositional.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("rand_seed", action='store_true')
    parser.add_argument("--show_results", action='store_true')
    parser.add_argument("--show_results_compositional", action='store_true')
    main(parser.parse_args())
