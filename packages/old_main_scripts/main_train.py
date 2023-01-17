import argparse
import sys
import pdb
import scipy

from packages.augmentation.diverse_augment import kmeans_augment
from packages.encoder_decoder_rnn.optimizer import LabelSmoothing
from packages.visualizations.visualize import visualize_losses
from packages.dataloading.morphology_dataloader import create_dataloader
from packages.dataloading.morphology_dataset import SigmorphonDataset
from packages.dataloading.vocab import MetaVocab, build_vocabulary, load_vocab
from packages.utils.util_functions import get_mask
from packages.utils.checkpointing import load_model, checkpoint_model
from packages.utils.constants import SIGM_DATA_PATH

from packages.transformer.self_attention_encoder import make_model
from packages.encoder_decoder_rnn.model import BidirectionalLemmaEncoder, TwoStepDecoderCell, make_inflector

import pdb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from typing import Optional

# NOTE: this should work for just morphological inflection. Reinflection is a bit more delicate.

BATCH_SIZE = 10
HIDDEN_DIM = 32
DEVICE = 'cuda'

def select_augmentation_points(strategy: str, augmentation_df: pd.DataFrame, num_hall: int) -> pd.DataFrame:
    if strategy == "random":
        sigm_hall_df = augmentation_df.sample(n=num_hall) # TODO: change this to use a specific strategy. 
        return sigm_hall_df
    elif strategy == "sample_by_uncertainty":
        # apply softmax, then pass those into pandas.
        r = 1
        norm_vector = augmentation_df["length"].pow(r)
        probs = scipy.special.softmax(augmentation_df['loss'] / norm_vector)
        sigm_hall_df = augmentation_df.sample(n=num_hall, weights=probs)
        return sigm_hall_df
    elif strategy == "sample_by_encoding_diversity": # TODO: use k-means++ to do the selection.
        # TODO implement
        sigm_hall_df = kmeans_augment(augmentation_df, num_hall) 
        return sigm_hall_df
    elif strategy == "sample_by_gradient_diversity":
        raise Exception("sampling by gradient embedding diversity is unimplemented")
    else:
        raise Exception("Unrecognized sampling strategy")
    
def get_dataloaders(num_hall: int, augmentation_strategy: str, vocab: MetaVocab, 
                    sigm_train_df: pd.DataFrame, sigm_val_df: pd.DataFrame, 
                    sigm_test_df: pd.DataFrame, sigm_hall_df: pd.DataFrame):
    # sigm_hall_df = pd.read_csv("results/spreadsheets/warm_start_model_aug_loss_unnormalized.csv")
    if num_hall > 0:
        assert sigm_hall_df is not None
        # sigm_hall_df = sigm_hall_df.sample(n=num_hall) # TODO: change this to use a specific strategy. 
        sigm_hall_df = select_augmentation_points(augmentation_strategy, sigm_hall_df, num_hall) # TODO: change this to use a specific strategy. 

    pure_train_length = len(sigm_train_df)
    train_sigm_df = pd.concat([sigm_train_df, sigm_hall_df])
    train_sigm_df['is_hallucinated'] = ([False] * pure_train_length) + ([True] * len(sigm_hall_df))
    sigm_val_df['is_hallucinated'] = ([False] * len(sigm_val_df)) 
    sigm_test_df['is_hallucinated'] = ([False] * len(sigm_test_df)) 
    train_sigm_dataset = SigmorphonDataset(train_sigm_df)
    val_sigm_dataset = SigmorphonDataset(sigm_val_df)
    test_sigm_dataset = SigmorphonDataset(sigm_test_df)
    train_dataloader = create_dataloader(train_sigm_dataset, BATCH_SIZE, vocab.vocab_char, vocab.vocab_tag, DEVICE)
    val_dataloader = create_dataloader(val_sigm_dataset, BATCH_SIZE, vocab.vocab_char, vocab.vocab_tag, DEVICE)
    test_dataloader = create_dataloader(test_sigm_dataset, BATCH_SIZE, vocab.vocab_char, vocab.vocab_tag, DEVICE)
    return train_dataloader, val_dataloader, test_dataloader

def get_data_augmentation_dataloader():
    sigm_hall_df = sigm_uncorrupted_w_hall_df.iloc[1000:]  
    # sigm_hall_df = pd.read_csv("results/spreadsheets/warm_start_model_aug_loss.csv")
    sigm_hall_df['is_hallucinated'] = ([True] * len(sigm_hall_df))
    hall_sigm_dataset = SigmorphonDataset(sigm_hall_df)
    hall_dataloader = create_dataloader(hall_sigm_dataset, BATCH_SIZE, vocab_char, 
                                        vocab_tag, DEVICE, shuffle=False, drop_last=False)
    return hall_dataloader 


def compute_loss(criterion, x, y, norm):
    """_summary_

    Args:
        x (torch.tensor): predictions. B x length of longest sequence in batch x vocab size
        y (torch.tensor): ground truth. B x length of longest sequence in batch 
        norm (torch.tensor): number of predictions being made. Size of each target, summed together

    Returns:
        _type_: _description_
    """
    # TODO: something wrong here.
    assert norm > 0
    loss_batch_size, loss_seq_len = x.shape[0], x.shape[1] 
    loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                        y.contiguous().view(-1)) # NOTE: I presume this broadcasts... should check if anything goes wrong
    
    return loss.sum() / norm 

# TODO: try dividing by log of length, or something that gives more weight to longer examples. 
def compute_loss_per_datapoint(criterion, x, y, norm_vec):
    """Computes the unnormalized loss per datapoint

    Args:
        x (torch.tensor): Predictions
        y (torch.tensor): Ground truth
        norm_vec (torch.tensor): Number of predictions that are made per item in the batch.

    Returns:
        Unnormalized loss. Vector with {x.shape[0]} elements.
    """
    loss_batch_size, loss_seq_len = x.shape[0], x.shape[1] 
    loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                        y.contiguous().view(-1)) # NOTE: I presume this broadcasts... should check if anything goes wrong
    loss = loss.view(loss_batch_size, loss_seq_len) 
    assert sum(loss[0][norm_vec[0]:]) == 0 # verifying that there's no loss for padding
    return loss.sum(axis=1) 

def run_train_epoch(train_dataloader, model, criterion, optimizer, vocab):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths, tag_lengths, is_hallucinated in train_dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)

        predictions = model(srcs, src_lengths, tags, vocab.bos_id, tag_lengths, tgts, tgt_lengths)
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
        norm = sum(tgt_lengths) - BATCH_SIZE 
        loss = compute_loss(criterion, predictions[:, :last_index - 1], tgts[:, 1:last_index], norm)  # -1 because we don't predict BOS

        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nb += 1

    return total_loss / nb

def run_val_epoch(dataloader, model, vocab, criterion):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths, tag_lengths, _ in dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)
        predictions = model(srcs, src_lengths, tags, vocab.bos_id, tag_lengths, tgts, tgt_lengths )
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        loss = compute_loss(criterion, predictions[:, :last_index -1], tgts[:, 1:last_index], sum(tgt_lengths) - BATCH_SIZE ) # TODO: make sure this is ok; bug prone line...

        total_loss += loss.data
        nb +=1
    return total_loss / nb

def test_model(dataloader, trained_model, vocab):
    actual_labels = []
    prediction_labels = []
    trained_model.eval()
    for srcs, tgts, tags, src_lengths, tgt_lengths, tag_lengths, _ in dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)
        prediction_dists = trained_model(srcs, src_lengths, tags, vocab.bos_id, tag_lengths)
        predictions = prediction_dists.max(dim=-1).indices # TODO: check this

        for i in range(predictions.shape[0]):
            prediction = vocab.vocab_char.lookup_tokens(predictions[i].tolist()) # TODO: just ignore everything after the BOS token.
            actual = vocab.vocab_char.lookup_tokens(tgts[i].tolist())
            if "</s>" in prediction:
                prediction_labels.append(prediction[:prediction.index("</s>")])
            else:
                prediction_labels.append(prediction)
            actual_labels.append(actual[1:actual.index("</s>")])
            print(f"Prediction: {prediction}")
            print(f"Actual: {actual}")
    total = len(actual_labels)
    assert total == len(prediction_labels)
    correct_arr = [1 if x == y else 0 for x, y in zip(actual_labels, prediction_labels)]
    num_correct = sum(correct_arr)
    print(f"Accuracy: {num_correct/total}")

def train_model(model, 
                train_dataloader, val_dataloader, nepochs, save_model_name,
                criterion, optimizer, vocab):
    for i in range(nepochs):
        model.train()
        epoch_loss = run_train_epoch(train_dataloader, model, criterion, optimizer, vocab)
        print(f"Current loss: {epoch_loss}") # TODO: replace with wandb

        model.eval()
        best_loss = float('inf')
        with torch.no_grad():
            val_loss = run_val_epoch(val_dataloader, model, vocab, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_model(model, fname=f"{save_model_name}.pt") 
            print(f"Current validation loss: {val_loss}") # TODO: replace with wandb
        
def visualize_loss_distribution(model, normalization_exp=1):
    load_model(model, "model_warm_start.pt")
    augmentation_dataloader = get_data_augmentation_dataloader()
    losses = []
    model.eval()
    src_text = []
    tgt_text = []
    all_tags = []
    lengths = []
    with torch.no_grad():
        for srcs, tgts, tags, src_lengths, tgt_lengths, tag_lengths, _ in augmentation_dataloader:
            # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)
            predictions = model(srcs, src_lengths, tags, bos_id, tgts, tgt_lengths)
            last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

            # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
            norm_vec = tgt_lengths - 1 
            loss = compute_loss_per_datapoint(predictions[:, :last_index -1], tgts[:, 1:last_index], norm_vec)  # -1 because we don't predict BOS
            losses.append(loss)
            lengths.append(norm_vec)

            for i in range(predictions.shape[0]):
                actual_src = ''.join(vocab_char.lookup_tokens(srcs[i].tolist()))
                actual_tgt = ''.join(vocab_char.lookup_tokens(tgts[i].tolist()))
                actual_src = actual_src[actual_src.index(">")+1:actual_src.index("/")-1]
                actual_tgt = actual_tgt[actual_tgt.index(">")+1:actual_tgt.index("/")-1]
                actual_tag = ';'.join(vocab_tag.lookup_tokens(tags[i].tolist()))
                actual_tag = actual_tag[actual_tag.index(">")+2:actual_tag.index("/")-2]
                all_tags.append(actual_tag)
                src_text.append(actual_src)
                tgt_text.append(actual_tgt)

    losses = torch.cat(losses)
    visualize_losses(losses)
    lengths = torch.cat(lengths)
    frame = pd.DataFrame({
        "src": src_text, 
        "tgt": tgt_text,
        "tag": all_tags,
        "length": lengths,
        "loss": losses
    })
    frame.to_csv(f"results/spreadsheets/warm_start_model_aug_loss_unnormalized.csv") # TODO: rename and save in scratch.
    return losses

def train_initial_model(language):
    """Train the initial model 
    """
    validation_frame= pd.read_csv(f"{SIGM_DATA_PATH}/{language}-dev", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    test_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-test", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    train_frame = pd.read_csv(f"{SIGM_DATA_PATH}/{language}-train-low", header=None, names=["src", "tgt" ,"tag"], sep='\t')
    sigm_hall_df_empty = pd.DataFrame(columns=train_frame.columns)
    vocab = load_vocab(pd.concat([train_frame, validation_frame, test_frame]), language)
    model = make_inflector(vocab.vocab_char, vocab.vocab_tag, vocab.src_padding_id, vocab.tag_padding_id, DEVICE, HIDDEN_DIM).to(DEVICE)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(0, 'NA', vocab, train_frame, validation_frame, test_frame, sigm_hall_df_empty)
    # criterion = nn.CrossEntropyLoss(ignore_index=vocab.padding_id, reduction='none', label_smoothing=0.1).to(DEVICE)
    # criterion = nn.CrossEntropyLoss(, ignore_index=vocab.padding_id, reduction='none').to(DEVICE)
    criterion = LabelSmoothing(len(vocab.vocab_char), vocab.src_padding_id, vocab.bos_id, vocab.unk_id).to(DEVICE) # NOTE: might have to change len(vocab)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(model, train_dataloader, val_dataloader, 10, f"{language}_initial", criterion, optimizer, vocab)
    load_model(model, fname=f"{language}_initial.pt")
    test_model(test_dataloader, model, vocab)

def main(args):
    if args.visualize_aug_loss_distribution:
        # TODO: compare the
        visualize_loss_distribution(model)
    elif args.train_initial_model:
        train_initial_model(args.language)
    elif args.evaluate_hall_subsample_strategy:
        nepochs = 10
        print(f"Training with {args.num_hall} augmented datapoints, using {args.subsample_strategy} strategy.")
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args.num_hall, args.subsample_strategy)
        load_model_name = f"model_warm_start"
        save_model_name = f"model_w_{args.num_hall}_aug_points_{args.subsample_strategy}"
        if args.use_warm_start:
            print("Using warm start")
            load_model(model, fname=f"{load_model_name}.pt")
        train_model(train_dataloader, val_dataloader, nepochs, save_model_name)
        load_model(model, fname=f"{save_model_name}.pt") 
        test_model(test_dataloader, model)


parser = argparse.ArgumentParser()
parser.add_argument("language", type=str) 
parser.add_argument("num_hall", type=int)
parser.add_argument("subsample_strategy", type=str)
parser.add_argument("--use_warm_start", action='store_true')
parser.add_argument("--visualize_aug_loss_distribution", action='store_true')
parser.add_argument("--train_initial_model", action='store_true')
main(parser.parse_args())