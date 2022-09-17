import argparse
import sys
import pdb
from packages.visualizations.visualize import visualize_losses
from packages.dataloading import vocab
from packages.dataloading.morphology_dataloader import create_dataloader
from packages.dataloading.morphology_dataset import SigmorphonDataset
from packages.dataloading.vocab import load_vocab
from packages.utils.util_functions import get_mask
from packages.utils.checkpointing import load_model, checkpoint_model

from packages.transformer.self_attention_encoder import make_model
from packages.encoder_decoder_rnn.model import BidirectionalLemmaEncoder, TwoStepDecoderCell, make_inflector

import pdb
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

# NOTE: this should work for just morphological inflection. Reinflection is a bit more delicate.

BATCH_SIZE = 10
DEVICE = 'cpu'
HIDDEN_DIM = 32

sigm_uncorrupted_w_hall_df = pd.read_csv("data/eng_1000_w_hall_train.tsv", sep='\t')
sigm_uncorrupted_df = sigm_uncorrupted_w_hall_df.iloc[0:1000]
vocab_char, vocab_tag = load_vocab(sigm_uncorrupted_df)
padding_id = vocab_char.get_stoi()["<blank>"]
bos_id = vocab_char.get_stoi()["<s>"]
eos_id = vocab_char.get_stoi()["</s>"]

def get_dataloaders(num_hall):
    # TODO: change this to load from our augmentation CSV instead.
    sigm_hall_df = sigm_uncorrupted_w_hall_df.iloc[1000:]  
    if num_hall == 0:
        sigm_hall_df = pd.DataFrame(columns=sigm_uncorrupted_df.columns)
    else:
        sigm_hall_df = sigm_hall_df.sample(n=num_hall) # TODO: can this be 0?

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    train_sigm_df, test_sigm_df= train_test_split(sigm_uncorrupted_df, test_size=1-train_ratio, random_state=0)
    val_sigm_df, test_sigm_df = train_test_split(test_sigm_df, test_size=test_ratio/(test_ratio + validation_ratio)) 

    pure_train_length = len(train_sigm_df)
    train_sigm_df = pd.concat([train_sigm_df, sigm_hall_df])
    train_sigm_df['is_hallucinated'] = ([False] * pure_train_length) + ([True] * len(sigm_hall_df))
    val_sigm_df['is_hallucinated'] = ([False] * len(val_sigm_df)) 
    test_sigm_df['is_hallucinated'] = ([False] * len(test_sigm_df)) 
    train_sigm_dataset = SigmorphonDataset(train_sigm_df)
    val_sigm_dataset = SigmorphonDataset(val_sigm_df)
    test_sigm_dataset = SigmorphonDataset(test_sigm_df)
    train_dataloader = create_dataloader(train_sigm_dataset, BATCH_SIZE, vocab_char, vocab_tag, DEVICE)
    val_dataloader = create_dataloader(val_sigm_dataset, BATCH_SIZE, vocab_char, vocab_tag, DEVICE)
    test_dataloader = create_dataloader(test_sigm_dataset, BATCH_SIZE, vocab_char, vocab_tag, DEVICE)
    return train_dataloader, val_dataloader, test_dataloader

def get_data_augmentation_dataloader():
    sigm_hall_df = sigm_uncorrupted_w_hall_df.iloc[1000:]  
    sigm_hall_df['is_hallucinated'] = ([True] * len(sigm_hall_df))
    hall_sigm_dataset = SigmorphonDataset(sigm_hall_df)
    hall_dataloader = create_dataloader(hall_sigm_dataset, BATCH_SIZE, vocab_char, 
                                        vocab_tag, DEVICE, shuffle=False, drop_last=False)
    return hall_dataloader 

model = make_inflector(vocab_char, vocab_tag, padding_id, HIDDEN_DIM).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=padding_id, reduction='none', label_smoothing=0.1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)

def compute_loss(x, y, norm):
    """_summary_

    Args:
        x (torch.tensor): predictions. B x length of longest sequence in batch x vocab size
        y (torch.tensor): ground truth. B x length of longest sequence in batch 
        norm (torch.tensor): number of predictions being made. Size of each target, summed together
        scale_factor (torch.tensor): scaling the sample differently based on whether it is hallucinated or not. 

    Returns:
        _type_: _description_
    """
    assert norm > 0
    loss_batch_size, loss_seq_len = x.shape[0], x.shape[1] 
    loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                        y.contiguous().view(-1)) # NOTE: I presume this broadcasts... should check if anything goes wrong
    loss = loss.view(loss_batch_size, loss_seq_len) 
    return loss.sum() / norm 

# TODO: try dividing by log of length, or something that gives more weight to longer examples. 
def compute_loss_per_datapoint(x, y, norm_vec):
    loss_batch_size, loss_seq_len = x.shape[0], x.shape[1] 
    loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                        y.contiguous().view(-1)) # NOTE: I presume this broadcasts... should check if anything goes wrong
    loss = loss.view(loss_batch_size, loss_seq_len) 
    return loss.sum(axis=1) / norm_vec 

def run_train_epoch(train_dataloader, model):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths, is_hallucinated in train_dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)

        predictions = model(srcs, src_lengths, tags, bos_id, tgts, tgt_lengths)
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
        norm = sum(tgt_lengths) - BATCH_SIZE 
        loss = compute_loss(predictions[:, :last_index -1], tgts[:, 1:last_index], norm)  # -1 because we don't predict BOS

        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nb += 1

    return total_loss / nb

def run_val_epoch(dataloader, model):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths, _ in dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)
        predictions = model(srcs, src_lengths, tags, bos_id, tgts, tgt_lengths)
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        loss = compute_loss(predictions[:, :last_index -1], tgts[:, 1:last_index], sum(tgt_lengths) - BATCH_SIZE ) # TODO: make sure this is ok; bug prone line...

        total_loss += loss.data
        nb +=1
    return total_loss / nb

def test_model(dataloader, trained_model):
    actual_labels = []
    prediction_labels = []
    trained_model.eval()
    for srcs, tgts, tags, src_lengths, tgt_lengths, _ in dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)
        prediction_dists = trained_model(srcs, src_lengths, tags, bos_id)
        # pdb.set_trace()
        predictions = F.softmax(prediction_dists, dim=-1).max(dim=-1).indices # TODO: check this


        for i in range(predictions.shape[0]):
            prediction = vocab_char.lookup_tokens(predictions[i].tolist()) # TODO: just ignore everything after the BOS token.
            actual = vocab_char.lookup_tokens(tgts[i].tolist())
            prediction_labels.append(prediction[:prediction.index("</s>")])
            actual_labels.append(actual[1:actual.index("</s>")])
            print(f"Prediction: {prediction}")
            print(f"Actual: {actual}")
    total = len(actual_labels)
    assert total == len(prediction_labels)
    correct_arr = [1 if x == y else 0 for x, y in zip(actual_labels, prediction_labels)]
    num_correct = sum(correct_arr)
    print(f"Accuracy: {num_correct/total}")


# character loss is easy to compute. Then, average those.
    # 1/n * (add the log likelihoods for each character) -- this is the confidence.
    # lower means less confident. 
# TODO: also compare with a random selection strategy.
# def choose_augmentation_samples(trained_model, gt_train_set, num_augmentations = 100): # ground truth train set.
#     # TODO: implement
#     pass

# Do we 
    # (a) train a new model with the augmentations?
    # (b) continue training on the previous model?

def train_model(train_dataloader, val_dataloader, nepochs, save_model_name):
    for i in range(nepochs):
        model.train()
        epoch_loss = run_train_epoch(train_dataloader, model)
        print(f"Current loss: {epoch_loss}")

        model.eval()
        best_loss = float('inf')
        with torch.no_grad():
            val_loss = run_val_epoch(val_dataloader, model)
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_model(model, fname=f"{save_model_name}.pt")
            print(f"Current validation loss: {val_loss}")
    # assume (a) for now.

# def train_with_augmentations(augmented_dataset): 
#     # TODO: implement
#     model = init_model()
#     train_dataloader = construct_dataloader(augmented_dataset)
#     train_model(model, train_dataloader, nepochs=30)
def visualize_loss_distribution(model):
    load_model(model, "model_warm_start.pt")
    augmentation_dataloader = get_data_augmentation_dataloader()
    losses = []
    model.eval()
    src_text = []
    tgt_text = []
    with torch.no_grad():
        for srcs, tgts, tags, src_lengths, tgt_lengths, _ in augmentation_dataloader:
            # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)
            predictions = model(srcs, src_lengths, tags, bos_id, tgts, tgt_lengths)
            last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

            # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
            norm_vec = tgt_lengths - 1 
            loss = compute_loss_per_datapoint(predictions[:, :last_index -1], tgts[:, 1:last_index], norm_vec)  # -1 because we don't predict BOS
            losses.append(loss)

            for i in range(predictions.shape[0]):
                actual_src = ''.join(vocab_char.lookup_tokens(srcs[i].tolist()))
                actual_tgt = ''.join(vocab_char.lookup_tokens(tgts[i].tolist()))
                actual_src = actual_src[actual_src.index(">")+1:actual_src.index("/")-1]
                actual_tgt = actual_tgt[actual_tgt.index(">")+1:actual_tgt.index("/")-1]
                src_text.append(actual_src)
                tgt_text.append(actual_tgt)

    losses = torch.cat(losses)
    visualize_losses(losses)
    frame = pd.DataFrame({
        "src": src_text, 
        "tgt": tgt_text,
        "loss": losses
    })
    frame.to_csv("results/spreadsheets/warm_start_model_aug_loss.csv")
    return losses

def main(args):
    if args.visualize_aug_loss_distribution:
        visualize_loss_distribution(model)
    else:
        nepochs = 30
        print(f"Training with {args.num_hall} augmented datapoints")
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(args.num_hall)
        load_model_name = f"model_warm_start"
        save_model_name = f"model_w_{args.num_hall}_aug_points"
        if args.use_warm_start:
            print("Using warm start")
            load_model(model, fname=f"{load_model_name}.pt")
        train_model(train_dataloader, val_dataloader, nepochs, save_model_name)
        load_model(model, fname=f"{save_model_name}.pt") 
        test_model(test_dataloader, model)

parser = argparse.ArgumentParser()
parser.add_argument("num_hall", type=int)
parser.add_argument("--use_warm_start", action='store_true')
parser.add_argument("--visualize_aug_loss_distribution", action='store_true')
main(parser.parse_args())