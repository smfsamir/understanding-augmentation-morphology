import pickle as pkl
import pdb

import numpy as np
import pandas as pd
from scipy.special import softmax

from .augment_selecter import AugmentationSelector 

class HighLossSampler(AugmentationSelector):
    """Samples losses with a higher loss. 
    """

    def __init__(self, likelihoods_path, initial_generation_frame, \
                 num_gold_test_examples, lengths, \
                 use_high_loss: bool): # TODO: add `lengths` implementation later. 
        AugmentationSelector.__init__(self, initial_generation_frame, num_gold_test_examples)
        # NOTE
        ## the indices in the example likelihoods correspond correspond to line numbers 
        ## in the test file that was used for producing the generations
        self.example_likelihoods = self._load_example_likelihoods(likelihoods_path)
        self.lengths = lengths 
        self.use_high_loss = use_high_loss
        # self.log_transform = log_transform
    
    def _load_example_likelihoods(self, likelihoods_path):
        with open(likelihoods_path, 'rb') as likelihoods_pkl:
             example_likelihoods = pkl.load(likelihoods_pkl)
        return example_likelihoods

    # NOTE: make sure you use the likelihoods of the sample rather than 
        # 
    def get_best_points(self, num_points: int) -> pd.DataFrame: 
        indices, log_likelihoods = zip(*self.example_likelihoods)
        indices_df = pd.DataFrame({
                'indices': indices
            }) 
        negative_log_likelihoods = -np.array(log_likelihoods)
        # assert all(x > 0 for x in scores)
        # if self.use_softmax_normalizer:
        #     weights = softmax(scores)
        # else:
        #     weights = scores
        # indices_df['weight'] = weights
        # subset_indices = indices_df.sample(n=num_points, weights='weight')['indices'].values
        # return self.get_augmentation_frame(subset_indices)
        subset_indices = (negative_log_likelihoods).argsort() 
        if self.use_high_loss:
            return self.get_augmentation_frame(np.array(indices)[subset_indices[-num_points:]])
        else:
            return self.get_augmentation_frame(np.array(indices)[subset_indices[:num_points]])