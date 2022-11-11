import os 
import pandas as pd
from .select_highest_loss import HighLossSampler
from .random_sampler import RandomSampler

def get_subset_selecter(language: str, augment_strategy: str, scratch_path_initial: str, 
                        initial_test_frame: pd.DataFrame):
    if augment_strategy == "uncertainty_sample":
        # TODO: construct the likelihood path. assert that it exists
        likelihood_pkl_path = f"{scratch_path_initial}/{language}_log_likelihoods.pickle"
        assert os.path.exists(likelihood_pkl_path) 
        return HighLossSampler(likelihood_pkl_path, initial_test_frame) 
    elif augment_strategy == "random":
        return RandomSampler(initial_test_frame)
    elif augment_strategy == "diversity_sample":
        pass
        # TODO: we need the embeddings for this.




    # elif augment_strategy == "diverse_sample":
    #     # TODO: implement later.
    #     return HighLossSampler(likelihood_pkl_path)