import re
import os
from packages.utils.constants import LANGUAGES, SCRATCH_PATH, ALGORITHMS
from packages.utils.util_functions import get_model_augment_path
from packages.augmentation.hyperparam_patterns import ALL_PATS

new_kwarg = ("train_medium", False)

def get_algorithm(model_dirname):
    for algorithm in ALGORITHMS:
        if model_dirname.startswith(algorithm):
            return algorithm
    raise Exception(f"This directory {model_dirname} doesn't seem to have an algorithm name")

for language in LANGUAGES:
    top_lang_dir= f"{SCRATCH_PATH}/{language}"
    for model_dir_name in os.listdir(top_lang_dir):
        if not os.path.isdir(f"{top_lang_dir}/{model_dir_name}"):
            continue
        new_kwarg_dict = {}
        if new_kwarg[0] in model_dir_name:
            print(f"Skipping {model_dir_name}")
            continue # already there
        else:
            new_kwarg_dict[new_kwarg[0]] = new_kwarg[1]
            for pat in ALL_PATS:
                re_search_res = re.search(pat, model_dir_name)
                if re_search_res:
                    re_key = re_search_res.group(0).split("=")[0]
                    re_val = re_search_res.group(0).split("=")[-1]
                    new_kwarg_dict[re_key] = re_val 
        new_model_dir_name = get_model_augment_path(language, get_algorithm(model_dir_name), **new_kwarg_dict) 
        print(f"The old dirname was {model_dir_name}; the new one is {new_model_dir_name}")
        src_path = f"{top_lang_dir}/{model_dir_name}"
        os.rename(src_path, new_model_dir_name)