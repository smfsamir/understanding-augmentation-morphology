from __future__ import annotations
import pdb
import torch
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from packages.encoder_decoder_rnn.model import BidirectionalLemmaEncoder, TwoStepDecoderCell, make_inflector
from packages.utils.constants import BATCH_SIZE, DEVICE
from packages.dataloading.morphology_dataset import SigmorphonDataset
from packages.dataloading.morphology_dataloader import create_dataloader
from packages.dataloading.vocab import load_vocab
from packages.utils.checkpointing import load_model

# TODO: fill in k-means clusters.
def get_embeddings(augmentation_dataframe, warm_start_model, vocab_char, vocab_tag):
    lemma_encoder = warm_start_model.lemma_encoder # extract from warm-started model; start wiht this.
    augmentation_dataframe['is_hallucinated'] = ([True] * len(augmentation_dataframe))
    hall_sigm_dataset = SigmorphonDataset(augmentation_dataframe)

    hall_dataloader = create_dataloader(hall_sigm_dataset, BATCH_SIZE, vocab_char, 
                                        vocab_tag, DEVICE, shuffle=False, drop_last=False)
    lemma_representations = []
    with torch.no_grad():
        for srcs, tgts, tags, src_lengths, tgt_lengths, _ in hall_dataloader:
            batch_size = srcs.shape[0]
            ranges = torch.arange(25).expand(batch_size, -1)
            cond = ranges < src_lengths.unsqueeze(-1)
            annotations_lemma, src_embeds_final = lemma_encoder(srcs, src_lengths) 
            annotations_w_zeros_padded = torch.where(cond.unsqueeze(-1), annotations_lemma, torch.zeros_like(annotations_lemma))
            annotations_sum = annotations_w_zeros_padded.sum(axis=1)
            mean_annotations = annotations_sum / src_lengths.unsqueeze(-1)
            lemma_representations.append(mean_annotations)
    all_lemma_representations = torch.cat(lemma_representations).numpy()
    all_lemma_representations = StandardScaler().fit_transform(all_lemma_representations)
    augmentation_rep_frame = pd.DataFrame(data= all_lemma_representations)
    augmentation_rep_frame['src'] = augmentation_dataframe['src']
    augmentation_rep_frame.to_csv("results/spreadsheets/augmentation_lemma_representations.csv")
    return augmentation_rep_frame
    # k_means_assignments = #... same size as lemma representations. save to same csv.

def obtain_kmeans_assignments():
    representation_frame = pd.read_csv("results/spreadsheets/augmentation_lemma_representations.csv", index_col=0)
    cols = list(map(str, range(32)))
    representations = representation_frame[cols]
    kmeans = KMeans(n_clusters = 100)
    labels = kmeans.fit_predict(representations)
    representation_frame['kmeans_label'] = labels
    representation_frame.to_csv("results/spreadsheets/augmentation_lemma_representations.csv")

# def get_gradient_embeddings():

def obtain_representations():
    augmentation_frame = pd.read_csv("results/spreadsheets/warm_start_model_aug_loss.csv")

    sigm_uncorrupted_w_hall_df = pd.read_csv("data/eng_1000_w_hall_train.tsv", sep='\t')
    sigm_uncorrupted_df = sigm_uncorrupted_w_hall_df.iloc[0:1000]
    vocab_char, vocab_tag = load_vocab(sigm_uncorrupted_df)
    padding_id = vocab_char.get_stoi()["<blank>"]
    bos_id = vocab_char.get_stoi()["<s>"]
    eos_id = vocab_char.get_stoi()["</s>"]
    HIDDEN_DIM = 32
    DEVICE = 'cpu'
    model = make_inflector(vocab_char, vocab_tag, padding_id, HIDDEN_DIM).to(DEVICE)
    load_model(model, "model_warm_start.pt")
    get_embeddings(augmentation_frame, model, vocab_char, vocab_tag)

def view_kmeans_assignments():
    kmeans_frame = pd.read_csv("results/spreadsheets/augmentation_lemma_representations.csv")[["kmeans_label", "src"]].sort_values(by='kmeans_label')
    kmeans_frame.to_csv("results/spreadsheets/augmentation_clusters_readable.csv")

def main(args):
    if args.obtain_representations:
        obtain_representations()
    elif args.obtain_kmeans_assignments:
        obtain_kmeans_assignments()
    elif args.view_kmeans_assignments:
        view_kmeans_assignments()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obtain_representations", action='store_true')
    parser.add_argument("--obtain_kmeans_assignments", action='store_true')
    parser.add_argument("--view_kmeans_assignments", action='store_true')
    main(parser.parse_args())