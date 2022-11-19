from typing import Dict
import numpy as np
import pdb
import math
import pandas as pd
import scipy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .augment_selecter import AugmentationSelector

class DiverseSampler(AugmentationSelector):

    def __init__(self, 
                embeddings_dict: Dict[int, np.array], 
                item_id_dict: Dict[int, np.array],
                token_ids_dict: Dict[int, np.array],
                src_dict: Dict[int, np.array],
                nclusters: int,
                initial_generation_frame: pd.DataFrame, num_gold_examples: int): 
        AugmentationSelector.__init__(self, initial_generation_frame, num_gold_examples)
        self.nclusters = nclusters
        self.embed_frame = self._construct_mean_pooled_embedding_frame(
            embeddings_dict, 
            item_id_dict,
            token_ids_dict, 
            src_dict
        )

    def _construct_mean_pooled_embedding_frame(self, 
                                               embeddings_dict: Dict[int, np.array], 
                                               item_id_dict: Dict[int, np.array],
                                               token_ids_dict: Dict[int, np.array],
                                               src_dict: Dict[int, np.array]):
        def _extract_mean_pooled_embed(tokens_enc, token_dict, unpooled_embeds):
            assert tokens_enc.ndim == 1
            def _is_valid_token(token_enc):
                if token_dict['symbols'][token_enc] in ['<pad>']:
                    return False
                return True

            valid_embeds = []
            for j in range(tokens_enc.shape[0]):
                if _is_valid_token(tokens_enc[j]):
                    valid_embeds.append(unpooled_embeds[j])
            return np.array(valid_embeds).mean(axis=0)

        embeddings = []
        orig_ids = []
        for seq_len in embeddings_dict.keys():
            embeds = np.array(embeddings_dict[seq_len])  # embed dim, number of datapoints, seq_len
            token_ids = np.array(token_ids_dict[seq_len]) # number of datapoints, seq_len
            item_ids = np.array(item_id_dict[seq_len])

            num_datapoints = token_ids.shape[0]
            for i in range(num_datapoints): # iterating over number of items
                embed = _extract_mean_pooled_embed(token_ids[i], src_dict.__dict__, embeds[:, i, :])
                embeddings.append(embed)
                orig_ids.append(item_ids[i])
        orig_ids = np.array(orig_ids)
        embeddings = np.array(embeddings)
        frame = pd.DataFrame(
            data=embeddings
        )
        frame["orig_id"] = orig_ids
        frame = frame[frame["orig_id"] >= self.num_gold_examples] # ensures that we only care to cluster the augmented points
        return frame

    def get_best_points(self, num_points):
        X = self.embed_frame[list(range(256))]
        X = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=self.nclusters)
        kmeans_labels = pd.Series(kmeans.fit_predict(X))
        kmeans_frame = pd.DataFrame({
            "orig_id": self.embed_frame["orig_id"].values, 
            "kmeans_label": kmeans_labels
        })
        # TODO: need to see if this works with some fake data..
        klabel_to_value = kmeans_frame['kmeans_label'].sample(num_points).value_counts().to_dict()
        sample_frames = []
        for klabel, num_samples in klabel_to_value.items():
            klabel_frame = kmeans_frame[kmeans_frame["kmeans_label"]==klabel]
            sample_frames.append(klabel_frame.sample(num_samples))
        diverse_frame = pd.concat(sample_frames)
        diverse_orig_ids = diverse_frame['orig_id'].values
        return self.get_augmentation_frame(diverse_orig_ids)