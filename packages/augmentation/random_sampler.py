import numpy as np
from .augment_selecter import AugmentationSelector 

class RandomSampler(AugmentationSelector):

    def __init__(self, initial_generation_frame): 
        AugmentationSelector.__init__(self, initial_generation_frame)
    
    # TODO: this assumes the augmented points are the last 10000 points
    def get_best_points(self, num_points): 
        return self.generation_frame[100:].sample(num_points)