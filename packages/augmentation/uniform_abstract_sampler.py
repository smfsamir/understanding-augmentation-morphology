import pdb
import numpy as np
import pandas as pd
import pickle as pkl
from .augment_selecter import AugmentationSelector 

# TODO: check for 100,000 instead of 10,000. Or, make it a parameter.
class UniformAbstractSampler(AugmentationSelector):
    """Sample abstract morphological templates.
    """

    def __init__(self, likelihoods_path, initial_generation_frame, \
            num_gold_test_examples, use_empirical, use_loss,
            aug_pool_size=10000): 
        AugmentationSelector.__init__(self, initial_generation_frame, num_gold_test_examples)
        self.example_likelihoods = self._load_example_likelihoods(likelihoods_path)
        self.use_empirical = use_empirical
        self.use_loss = use_loss
        self.aug_pool_size = aug_pool_size
    
    def _load_example_likelihoods(self, likelihoods_path):
        with open(likelihoods_path, 'rb') as likelihoods_pkl:
            example_likelihoods = pkl.load(likelihoods_pkl)
        return example_likelihoods

    def _build_augmentation_frame(self) -> pd.DataFrame:
        """Build a DataFrame with columns 
        |src|tag|nll|tgt|. The rows should be sorted by NLL (descending).

        Returns:
            pd.DataFrame: _description_
        """
        indices, log_likelihoods = zip(*self.example_likelihoods)
        negative_log_likelihoods = -np.array(log_likelihoods)
        likelihood_frame = pd.DataFrame({
            "nll": negative_log_likelihoods
        }, index = indices)
        num_generation = len(self.generation_frame)
        num_gold = num_generation - self.aug_pool_size
        augmentation_frame = self.generation_frame.loc[np.arange(num_gold, num_generation)]
        augmentation_frame = augmentation_frame.join(likelihood_frame)
        assert len(augmentation_frame) == self.aug_pool_size 
        if self.use_loss:
            augmentation_frame = augmentation_frame.sort_values(by='nll', ascending=False)
        else:
            augmentation_frame = augmentation_frame.sample(frac=1.0, replace=False)
        return augmentation_frame

    def get_best_points(self, num_points: int) -> pd.DataFrame:
        augmentation_frame = self._build_augmentation_frame()
        subset_frames = []
        for _ in range(num_points):
            next_point_row_index = self._get_next_point(augmentation_frame, self.use_empirical)
            assert next_point_row_index in augmentation_frame.index.values, f"{next_point_row_index} is not in the frame"
            try:
                subset_frames.append(augmentation_frame.loc[[next_point_row_index]])
            except IndexError as e:
                print(f"{next_point_row_index} caused an error")
                raise e
            augmentation_frame = augmentation_frame.drop(index=next_point_row_index)
        subset_frame = pd.concat(subset_frames)
        return self.get_augmentation_frame(subset_frame.index.values)

    def _get_next_point(self, augmentation_frame: pd.DataFrame,
                              sample_empirical: bool) -> int:
        """Get the next point that is sampled. It should then be removed from the pool of
        options.

        Args:
            augmentation_frame (pd.DataFrame): Pandas DataFrame with columns 
                |tag|NLL|. A major assumption is that the rows are sorted in descending
                order by the NLL.
            augmentation_frame (pd.DataFrame): Pandas DataFrame.

        Returns:
            int: An index into the Augmentation DataFrame, representing which point should be taken next
            (and thus made unavailable for the next round of sampling).
        """
        if sample_empirical:
            morph_tag = augmentation_frame['tag'].sample(n=1).values[0]
        else: # sample uniformly.
            morph_tag = augmentation_frame['tag'].drop_duplicates().sample(n=1).values[0]
        subset_tag_frame = augmentation_frame[augmentation_frame['tag'] == morph_tag]
        return subset_tag_frame.index.values[0]