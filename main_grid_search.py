import argparse
import subprocess
from enum import Enum
from itertools import product
from packages.utils.util_functions import get_model_augment_path
from packages.utils.constants import LANGUAGES
import os

# TODO: delete later; in util_functions
def generate_hyperparams(hyperparam_dict):
    keys = hyperparam_dict.keys()
    values = hyperparam_dict.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))
    
def hparam_comb_tested(hparam_comb, curr_language, augmentation_type):
    augment_path = get_model_augment_path(curr_language, augmentation_type, **hparam_comb)
    if os.path.exists(f"{augment_path}/final_results.txt"):
        print(f"{augment_path} has already been generated; skipping")
        return True
    else:
        return False

def build_argument_list(hparam_comb):
    argument_list = []

    # required arguments
    argument_list.append(str(hparam_comb['num_aug']))

    # optional arguments
    for hparam in hparam_comb:
        if hparam == 'r':
            argument_list.extend(['--r', str(hparam_comb['r'])])
        elif hparam == 'train_medium':
            if hparam_comb['train_medium']:
                argument_list.append('--train_medium')
        elif hparam == 'use_softmax_normalizer':
            if hparam_comb['use_softmax_normalizer']:
                argument_list.append('--use_softmax_normalizer')
        elif hparam == 'use_high_loss':
            if hparam_comb['use_high_loss']:
                argument_list.append('--use_high_loss')
        elif hparam == 'num_aug':
            pass # handled in required arguments
        elif hparam == 'use_empirical':
            if hparam_comb['use_empirical']:
                argument_list.append('--use_empirical')
        elif hparam == 'use_loss':
            if hparam_comb['use_loss']:
                argument_list.append('--use_loss')
        elif hparam == 'rand_seed':
            continue
        elif hparam == 'aug_pool_size':
            continue
        else:
            raise Exception(f"Unrecognized hyperparameter {hparam}")
    return argument_list

def grid_search_random(language: str, rand_seed: int, aug_pool_size: int):
    hyperparams = {
        "num_aug": [128, 256, 512, 1024, 2048],
        "train_medium": [False], 
        "rand_seed": [rand_seed],
        "aug_pool_size": [aug_pool_size]
    }
    for hparam_comb in generate_hyperparams(hyperparams):
        if hparam_comb_tested(hparam_comb, language, 'random'):
            continue
        else:
            argument_list = build_argument_list(hparam_comb)
            print(f"Running random pipeline with argument list {argument_list}")
            result = subprocess.run(["python", "main_transformer.py", language, str(rand_seed), "random", str(aug_pool_size)] + argument_list + ['--run_random_sampling_pipeline'])
            print(result)
            if not result.returncode == 0:
                raise Exception("Tried running the random sampling pipeline but some step failed.")
            
def grid_search_initial(language: str, rand_seed: int, aug_pool_size: int):
    hyperparams = {
        "train_medium": [False], 
        "rand_seed": [rand_seed], 
        "aug_pool_size": [aug_pool_size]
    }

    for hparam_comb in generate_hyperparams(hyperparams):
        if hparam_comb_tested(hparam_comb, language, 'initial'):
            continue
        result = subprocess.run(["python", "main_transformer.py", language, str(rand_seed), "initial",  str(aug_pool_size), str(0),] + ['--run_initial_pipeline'], check=True)
        print(result)
        if not result.returncode == 0:
            raise Exception("Tried running the random sampling pipeline but some step failed.")

def grid_search_uncertainty(language, rand_seed: int, aug_pool_size: int):
    # TODO: each of these needs functions to turn them into options
    hyperparams = {
        "train_medium": [False],
        "num_aug": [128, 256, 512, 1024, 2048],
        "use_high_loss": [True, False], 
        "rand_seed": [rand_seed], 
        "aug_pool_size": [aug_pool_size]
    }
    for hparam_comb in generate_hyperparams(hyperparams):
        if hparam_comb_tested(hparam_comb, language, 'uncertainty_sample'):
            continue
        else:
            argument_list = build_argument_list(hparam_comb)
            print(f"Running uncertainty pipeline with argument list {argument_list} for {language}")
            result = subprocess.run(["python", "main_transformer.py", language, str(rand_seed), \
                "uncertainty_sample", str(aug_pool_size)] + argument_list + ['--run_uncertainty_sampling_pipeline'])
            print(result)
            if not result.returncode == 0:
                raise Exception("Tried running the pipeline but some step failed")

def grid_search_uat(language: str, rand_seed: int, aug_pool_size: int):
    hyperparams = {
        "train_medium": [False], 
        "num_aug": [128, 256, 512, 1024, 2048],
        "use_empirical": [False],
        "use_loss": [True], 
        "rand_seed": [rand_seed],
        "aug_pool_size": [aug_pool_size]
    }
    for hparam_comb in generate_hyperparams(hyperparams):
        if hparam_comb_tested(hparam_comb, language, 'uat'):
            continue
        else:
            argument_list = build_argument_list(hparam_comb)
            print(f"Running uat pipeline with argument list {argument_list} for {language}")
            result = subprocess.run(["python", "main_transformer.py", language, str(rand_seed), \
                "uat", str(aug_pool_size)] + argument_list + ['--run_uat_pipeline'], check=True)
            print(result)
            if not result.returncode == 0:
                raise Exception("Tried running the pipeline but some step failed")

def main(args):
    for iter_language in LANGUAGES:
    # for iter_language in ['bengali']:
        if args.train_initial:
            grid_search_initial(iter_language, args.rand_seed, args.aug_pool_size)
        elif args.train_uncertainty:
            grid_search_uncertainty(iter_language, args.rand_seed, args.aug_pool_size)
        elif args.train_uat:
            grid_search_uat(iter_language, args.rand_seed, args.aug_pool_size )
        elif args.train_random:
            grid_search_random(iter_language, args.rand_seed, args.aug_pool_size)


parser = argparse.ArgumentParser()
parser.add_argument("rand_seed", type=int)
parser.add_argument("aug_pool_size", type=int)
parser.add_argument("--train_initial", action='store_true')
parser.add_argument("--train_uncertainty", action='store_true')
parser.add_argument("--train_uat", action='store_true')
parser.add_argument("--train_random", action='store_true')
main(parser.parse_args())
