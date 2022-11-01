import pandas as pd
from scipy.special import softmax
import pickle
from .augment_selecter import AugmentSelector

class HighLossSampler(AugmentSelector):

    def __init__(self, likelihoods_path, lengths):
        AugmentSelector.__init__()
        # NOTE
        ## the indices in the example likelihoods correspond correspond to line numbers 
        ## in the test file that was used for producing the generations
        self.example_likelihoods = self._load_example_likelihoods(likelihoods_path)
        self.lengths = lengths 
    
    def _load_example_likelihoods(self, likelihoods_path):
        with open(likelihoods_path, 'rb') as likelihoods_pkl:
             example_likelihoods = pkl.load(encoder_embeds_pkl)
        return example_likelihoods

    def get_best_points(self, num_points): # returns an array comprising samples from the full augmentation dataset
        indices, log_likelihoods = zip(*self.example_likelihoods)
        indices_series = pd.Series(indices) 
        probs = softmax(log_likelihoods)
        return indices.sample(n=num_points)