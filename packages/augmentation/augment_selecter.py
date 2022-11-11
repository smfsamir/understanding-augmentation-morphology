import numpy as np
import pandas as pd

class AugmentationSelector:

    def __init__(self, initial_test_frame: pd.DataFrame):
        self.generation_frame = initial_test_frame
        pass
    
    def get_augmentation_frame(self, indices: np.array):
        return self.generation_frame.iloc[indices]

    def get_best_points(self, num_points):
        raise NotImplementedError()

