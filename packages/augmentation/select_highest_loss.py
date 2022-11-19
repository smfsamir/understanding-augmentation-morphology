import numpy as np
import pandas as pd
from scipy.special import softmax
import pickle as pkl

from .augment_selecter import AugmentationSelector 

class HighLossSampler(AugmentationSelector):

    def __init__(self, likelihoods_path, initial_generation_frame, num_gold_test_examples, lengths=None): # TODO: add `lengths` implementation later. 
        AugmentationSelector.__init__(self, initial_generation_frame, num_gold_test_examples)
        # NOTE
        ## the indices in the example likelihoods correspond correspond to line numbers 
        ## in the test file that was used for producing the generations
        self.example_likelihoods = self._load_example_likelihoods(likelihoods_path)
        self.lengths = lengths 
    
    def _load_example_likelihoods(self, likelihoods_path):
        with open(likelihoods_path, 'rb') as likelihoods_pkl:
             example_likelihoods = pkl.load(likelihoods_pkl)
        return example_likelihoods

    # NOTE: make sure you use the likelihoods of the sample rather than 
        # 
    def get_best_points(self, num_points): 
        # TODO: we really need to test this. are these indeed the datapoints with the highest loss?
            # TODO: a basic test is to check that the subset indices are all > 99. 
            # TODO: a possible problem is that the likelihoods are log likelihoods. We want *negative* log likelihoods.
        indices, log_likelihoods = zip(*self.example_likelihoods)
        indices_series = pd.Series(indices) 
        negative_log_likelihoods = -np.array(log_likelihoods)
        probs = softmax(negative_log_likelihoods)
        subset_indices = indices_series.sample(n=num_points, weights=probs).values
        return self.get_augmentation_frame(subset_indices)