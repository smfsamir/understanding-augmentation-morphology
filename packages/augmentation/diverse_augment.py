import pdb
import math
import pandas as pd

def kmeans_augment(augmentation_frame: pd.DataFrame, num_hall: int):
    kmeans_frame = pd.read_csv("results/spreadsheets/augmentation_lemma_representations.csv")[["kmeans_label", "src"]]
    num_clusters = len(kmeans_frame['kmeans_label'].unique())
    num_per_cluster = math.ceil(num_hall / num_clusters) 
    augment_samples_frame = kmeans_frame.merge(augmentation_frame, left_on="src", right_on="src") # NOTE: this merge can create duplicates if the og augmentation frame has duplicates
    augment_samples_frame = augment_samples_frame.groupby('kmeans_label').apply(lambda x: x.nlargest(n=num_per_cluster, columns='loss')) # NOTE changing to highest likelihood instead.
    augment_samples_frame = augment_samples_frame.sample(n=num_hall) 
    assert len(augment_samples_frame) == num_hall
    return augment_samples_frame[["src", "tgt", "tag"]]