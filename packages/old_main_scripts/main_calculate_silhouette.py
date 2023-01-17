from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from packages.utils.constants import SCRATCH_PATH
from packages.augmentation.subset_selecter_strategy import get_subset_selecter
from packages.fairseq_utils.dataloading_utils import get_initial_generation_frame

def calculate_silhouette():
    diverse_sampler = get_subset_selecter("bengali", "diversity_sample", f"{SCRATCH_PATH}/bengali/initial", get_initial_generation_frame("bengali"))
    embed_frame = diverse_sampler.embed_frame
    embeds = embed_frame[list(range(256))]
    k_to_silh_score = {}
    for k in range(2,20):
        kmeans = KMeans(k)
        cluster_labels = kmeans.fit_predict(embeds)
        score = silhouette_score(embeds, cluster_labels)
        k_to_silh_score[k] = score
    print("Obtained the following silhouette scores for Bengali")
    print(k_to_silh_score)

if __name__ == "__main__":
    calculate_silhouette()