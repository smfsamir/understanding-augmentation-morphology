from typing import Dict
import numpy as np
import pdb
import math
import pandas as pd
import scipy
import pandas as pd
from sklearn.cluster import KMeans

from .augment_selecter import AugmentationSelector

# def kmeans_augment(augmentation_frame: pd.DataFrame, num_hall: int):
#     kmeans_frame = pd.read_csv("results/spreadsheets/augmentation_lemma_representations.csv")[["kmeans_label", "src"]]
#     num_clusters = len(kmeans_frame['kmeans_label'].unique())
#     num_per_cluster = math.ceil(num_hall / num_clusters) 
#     augment_samples_frame = kmeans_frame.merge(augmentation_frame, left_on="src", right_on="src") # NOTE: this merge can create duplicates if the og augmentation frame has duplicates
#     # augment_samples_frame = augment_samples_frame.groupby('kmeans_label').apply(lambda x: x.sample(n=num_per_cluster)) # NOTE changing to highest likelihood instead.
#     augment_samples_frame = augment_samples_frame.groupby('kmeans_label').apply(lambda x: x.sample(n=num_per_cluster, weights=scipy.special.softmax(x['loss']))) # NOTE changing to highest likelihood instead.
#     augment_samples_frame = augment_samples_frame.sample(n=num_hall) 
#     assert len(augment_samples_frame) == num_hall
#     return augment_samples_frame[["src", "tgt", "tag"]]

class DiverseSampler(AugmentationSelector):

    def __init__(self, 
                embeddings_dict: Dict[int, np.array], 
                item_id_dict: Dict[int, np.array],
                token_ids_dict: Dict[int, np.array],
                src_dict: Dict[int, np.array],
                nclusters: int,
                initial_generation_frame: pd.DataFrame): 
        AugmentationSelector.__init__(self, initial_generation_frame)
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
        return frame

    def get_best_points(self, num_points):
        # TODO: implement
        X = self.embed_frame[list(range(256))]
        kmeans = KMeans(n_clusters=self.nclusters)
        kmeans_labels = pd.Series(kmeans.fit_predict(X))
        kmeans_frame = pd.DataFrame(
            "orig_id": self.embed_frame["orig_id"].values, 
            "kmeans_label": kmeans_labels
        )
        # TODO: need to see if this works with some fake data..
        klabel_to_value = kmeans_frame['kmeans_label'].sample(num_points).value_counts().to_dict()
        diverse_sample_frame = kmeans_frame.groupby("kmeans_label").apply(lambda k_frame: k_frame.sample(n=klabel_to_value[k_frame.get_group('kmeans_label')]))
        orig_ids = diverse_sample_frame['orig_id'].values
        return self.get_augmentation_frame(orig_ids)


        # if i group by, i know 
