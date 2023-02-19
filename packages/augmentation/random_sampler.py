import pandas as pd
import numpy as np
from .augment_selecter import AugmentationSelector 

class RandomSampler(AugmentationSelector):

    def __init__(self, initial_generation_frame: pd.DataFrame, num_gold_examples: int): 
        AugmentationSelector.__init__(self, initial_generation_frame, num_gold_examples)
    
    # NOTE: this assumes the augmented points are the bottom part of the frame
    def get_best_points(self, num_points): 
        return self.generation_frame[self.num_gold_examples: ].sample(num_points)