import sys
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
HALL_SCALE_FACTOR = 0.95
sigm_uncorrupted_w_hall_df= pd.read_csv("data/eng_1000_w_hall_train.tsv", sep='\t')
sigm_uncorrupted_df = sigm_uncorrupted_w_hall_df.iloc[0:1000]
sigm_hall_df = sigm_uncorrupted_w_hall_df.iloc[1000:] 
vocab_char, vocab_tag = load_vocab(sigm_uncorrupted_df)
padding_id = vocab_char.get_stoi()["<blank>"]
bos_id = vocab_char.get_stoi()["<s>"]
eos_id = vocab_char.get_stoi()["</s>"]

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
batch_size = 10
device = 'cuda'
train_dataloader = create_dataloader(train_sigm_dataset, batch_size, vocab_char, vocab_tag, device)
val_dataloader = create_dataloader(val_sigm_dataset, batch_size, vocab_char, vocab_tag, device)
test_dataloader = create_dataloader(test_sigm_dataset, batch_size, vocab_char, vocab_tag, device)



hidden_dim = 32
# embed_layer = nn.Embedding(len(vocab_char) + 1, embedding_dim=hidden_dim, padding_idx=padding_id) # +1 for padding token?
# tag_encoder = make_model(len(vocab_tag), d_model=hidden_dim)
# lemma_encoder = BidirectionalLemmaEncoder(embed_layer, hidden_dim)
# decoder = TwoStepDecoderCell(embed_layer, hidden_dim, len(vocab_char))
model = make_inflector(vocab_char, vocab_tag, padding_id, hidden_dim).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=padding_id, reduction='none').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# TODO: make train and validation loop.
# TODO: we should also try generating an example.

def compute_loss(x, y, norm, scale_factor):
    """_summary_

    Args:
        x (torch.tensor): predictions. B x length of longest sequence in batch x vocab size
        y (torch.tensor): ground truth. B x length of longest sequence in batch 
        norm (torch.tensor): number of predictions being made. Size of each target, summed together
        scale_factor (torch.tensor): scaling the sample differently based on whether it is hallucinated or not. 

    Returns:
        _type_: _description_
    """
    # TODO: scale the loss as needed. 
    assert norm > 0
    loss_batch_size, loss_seq_len = x.shape[0], x.shape[1] 
    loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                        y.contiguous().view(-1)) # NOTE: I presume this broadcasts... should check if anything goes wrong
    loss = loss.view(loss_batch_size, loss_seq_len) * scale_factor[:, None]
    return loss.sum() / norm 

scale_factor_regular = torch.ones(batch_size, device=device)
scale_factor_hall = torch.ones(batch_size, device=device) * HALL_SCALE_FACTOR
def run_train_epoch(train_dataloader, model):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths, is_hallucinated in train_dataloader:
        # srcs, src_lengths, tags, tgts, tgt_lengths = srcs.to(device), src_lengths.to(device), tags.to(device), tgts.to(device), tgt_lengths.to(device)

        predictions = model(srcs, src_lengths, tags, bos_id, tgts, tgt_lengths)
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
        scale_factor = torch.where(is_hallucinated, scale_factor_hall, scale_factor_regular)
        norm = sum(scale_factor* tgt_lengths) - sum(scale_factor)
        loss = compute_loss(predictions[:, :last_index -1], tgts[:, 1:last_index], norm, scale_factor)  # -1 because we don't predict BOS

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

        # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
        loss = compute_loss(predictions[:, :last_index -1], tgts[:, 1:last_index], sum(tgt_lengths) - batch_size, scale_factor_regular) # TODO: make sure this is ok; bug prone line...

        total_loss += loss.data
        nb +=1
    return total_loss / nb

def test_model(dataloader, trained_model):
    actual_labels = []
    prediction_labels = []
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

# TODO: load tensorboard
# for i in range(30):
#     model.train()
#     epoch_loss = run_train_epoch(train_dataloader, model)
#     print(f"Current loss: {epoch_loss}")


#     model.eval()
#     best_loss = float('inf')
#     with torch.no_grad():
#         val_loss = run_val_epoch(val_dataloader, model)
#         if val_loss < best_loss:
#             best_loss = val_loss
#             checkpoint_model(model)
#         print(f"Current validation loss: {val_loss}")

load_model(model) # mutation
test_model(test_dataloader, model)