import numpy as np
import pandas as pd

class AugmentationSelector:

    def __init__(self, initial_test_frame: pd.DataFrame, num_gold_examples: int):
        self.generation_frame = initial_test_frame
        self.num_gold_examples = num_gold_examples
        pass
    
    def get_augmentation_frame(self, indices: np.array):
        return self.generation_frame.loc[indices]

    def get_best_points(self, num_points: int) -> pd.DataFrame:
        raise NotImplementedError()