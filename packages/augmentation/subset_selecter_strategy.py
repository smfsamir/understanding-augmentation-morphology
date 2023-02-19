import pdb
import pickle
import os 
import pandas as pd
from typing import Dict
from .select_highest_loss import HighLossSampler
from .random_sampler import RandomSampler
from .diverse_sampler import DiverseSampler
from .uniform_abstract_sampler import UniformAbstractSampler

def _load_initial_representations_bundle(scratch_path_initial, language):
    # NOTE: the "_true_initial" suffix is necessary due to the way we constructed the embeddings.
    with open(f"{scratch_path_initial}/{language}_embeddings_true_initial.pickle", "rb") as embeddings_pkl, \
         open(f"{scratch_path_initial}/{language}_src_tokens_true_initial.pickle", "rb") as token_ids_pkl, \
         open(f"{scratch_path_initial}/{language}_ids_true_initial.pickle", "rb") as item_ids_pkl, \
         open(f"{scratch_path_initial}/{language}_src_dict_true_initial.pickle", "rb") as src_dict_pkl:

        embeddings_dict = pickle.load(embeddings_pkl)
        item_id_dict = pickle.load(item_ids_pkl)
        token_ids_dict = pickle.load(token_ids_pkl)
        src_dict = pickle.load(src_dict_pkl)
    return embeddings_dict, item_id_dict, token_ids_dict, src_dict

def get_subset_selecter(language: str, augment_strategy: str, scratch_path_initial: str, 
                        initial_test_frame: pd.DataFrame, 
                        num_gold_test_examples: int,
                        lengths: pd.Series,
                        **kwargs: Dict):
    if augment_strategy == "uncertainty_sample":
        # TODO: construct the likelihood path. assert that it exists
        likelihood_pkl_path = f"{scratch_path_initial}/{language}_log_likelihoods.pickle"
        use_high_loss = kwargs['use_high_loss']
        assert os.path.exists(likelihood_pkl_path), f"Couldn't find likelihood file at {likelihood_pkl_path}"
        return HighLossSampler(likelihood_pkl_path, initial_test_frame, num_gold_test_examples,  lengths, use_high_loss)
    elif augment_strategy == "random":
        return RandomSampler(initial_test_frame, num_gold_test_examples)
    elif augment_strategy == "uat":
        likelihood_pkl_path = f"{scratch_path_initial}/{language}_log_likelihoods.pickle"
        use_empirical = kwargs['use_empirical']
        use_loss = kwargs['use_loss']
        aug_pool_size = kwargs['aug_pool_size']
        return UniformAbstractSampler(likelihood_pkl_path, initial_test_frame, num_gold_test_examples, use_empirical, use_loss, aug_pool_size)
    elif augment_strategy == "diversity_sample":
        embeddings_dict, item_id_dict, token_ids_dict, src_dict = _load_initial_representations_bundle(scratch_path_initial, language)
        k = kwargs['k']
        return DiverseSampler(embeddings_dict, item_id_dict, token_ids_dict, src_dict, k, initial_test_frame, num_gold_test_examples) 
        # TODO: we need the embeddings for this.
    else:
        raise Exception(f"No strategy {augment_strategy}")



    # elif augment_strategy == "diverse_sample":
    #     # TODO: implement later.
    #     return HighLossSampler(likelihood_pkl_path)