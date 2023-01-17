import datetime
from itertools import product

import pandas as pd
from .constants import SIGM_DATA_PATH, SCRATCH_PATH, FAIRSEQ_SCRIPTS_PATH, INITIAL_MODEL_PARAMS

def map_list(func, arr):
    return list(map(func, arr))

def get_mask(tensors, mask_token):
    """Return a mask that is the same shape as tensors. True if entry at tensors matches the mask token.

    Args:
        tensors (torch.Tensor): _description_
        mask_token (int): _description_
    """
    return tensors == mask_token

def get_model_augment_path(language, augmentation_type, **kwargs):
    """

    Args:
        language (str): Language the model is being trained for.
        augmentation_type (str): Augmentation type for the model. 
    """
    kwarg_strs= [f"{k}={v}" for k, v in kwargs.items()]
    kwarg_strs.sort()
    kwarg_str = "_".join(kwarg_strs)
    path = f"{SCRATCH_PATH}/{language}/{augmentation_type}_{kwarg_str}"
    return path 

def get_number_test_examples(language, **kwargs):
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    return len(test_frame)

def get_initial_model_path(language, **kwargs):
    init_kwargs = {k: kwargs[k] for k in INITIAL_MODEL_PARAMS}
    initial_augment_path = get_model_augment_path(language, 'initial', **init_kwargs)
    return initial_augment_path

def generate_hyperparams(hyperparam_dict):
    keys = hyperparam_dict.keys()
    values = hyperparam_dict.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))

def load_gold_train_validation_test(language, train_medium: bool):
    if train_medium:
        train_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-train-medium")
    else: 
        train_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-train-low")
    validation_frame= load_sigm_file(f"{SIGM_DATA_PATH}/{language}-dev")
    test_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-test")
    return train_frame, validation_frame, test_frame

def load_sigm_file(path):
    return pd.read_csv(path, header=None, names=["src", "tgt" ,"tag"], sep='\t')

def tokenize_row_src(row):
    tokens = list(row.src) + row.tag.split(";")
    return " ".join(tokens)

def tokenize_row_tgt(row):
    tokens = list(row.tgt)
    return " ".join(tokens)

def construct_cg_test_set(language, data_quantity):
    dev_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-dev")

    low_train_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-train-low")
    medium_train_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-train-medium")
    high_train_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-train-high")
    if data_quantity == 'low':
        exclude_frame = pd.concat([low_train_frame, dev_frame])
    elif data_quantity == 'medium':
        exclude_frame = pd.concat([medium_train_frame, dev_frame])
    exclude_frame_lemmas = set(exclude_frame['src'].values)
    test_frame = load_sigm_file(f"{SIGM_DATA_PATH}/{language}-test")
    all_frames = pd.concat([low_train_frame, medium_train_frame, high_train_frame, dev_frame, test_frame])
    cg_test_frame = all_frames[~all_frames['src'].isin(exclude_frame_lemmas)]
    all_frame_len = len(all_frames)
    cg_frame_len = len(cg_test_frame)
    print(f"Out of the {all_frame_len} entries, we took {cg_frame_len} of them")
    return cg_test_frame
