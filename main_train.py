from packages.dataloading import vocab
from packages.dataloading.morphology_dataloader import create_dataloader
from packages.dataloading.morphology_dataset import SigmorphonDataset
from packages.dataloading.vocab import load_vocab
from packages.utils.util_functions import get_mask

from packages.transformer.self_attention_encoder import make_model
from packages.encoder_decoder_rnn.model import BidirectionalLemmaEncoder, TwoStepDecoderCell, make_inflector

import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch

# NOTE: this should work for just morphological inflection. Reinflection is a bit more delicate.
sigm_df = pd.read_csv("data/eng_1000_train.tsv", sep='\t')
vocab_char, vocab_tag = load_vocab(sigm_df)
padding_id = vocab_char.get_stoi()["<blank>"]

# TODO: make train/validation split
train_sigm_df, val_sigm_df = train_test_split(sigm_df, train_size=0.8, random_state=0)

train_sigm_dataset = SigmorphonDataset(train_sigm_df)
val_sigm_dataset = SigmorphonDataset(val_sigm_df)
batch_size = 32
train_dataloader = create_dataloader(train_sigm_dataset, batch_size, vocab_char, vocab_tag, 'cpu')
val_dataloader = create_dataloader(val_sigm_dataset, batch_size, vocab_char, vocab_tag, 'cpu')



hidden_dim = 64
# embed_layer = nn.Embedding(len(vocab_char) + 1, embedding_dim=hidden_dim, padding_idx=padding_id) # +1 for padding token?
# tag_encoder = make_model(len(vocab_tag), d_model=hidden_dim)
# lemma_encoder = BidirectionalLemmaEncoder(embed_layer, hidden_dim)
# decoder = TwoStepDecoderCell(embed_layer, hidden_dim, len(vocab_char))
model = make_inflector(vocab_char, vocab_tag, padding_id, hidden_dim)
criterion = nn.CrossEntropyLoss(ignore_index=padding_id, reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.01)

# TODO: make train and validation loop.
# TODO: we should also try generating an example.

def compute_loss(x, y, norm):
    assert norm > 0
    loss = criterion(x.contiguous().view(-1, x.size(-1)), 
                        y.contiguous().view(-1)) / norm 
    return loss

def run_train_epoch(train_dataloader, model):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths in train_dataloader:
        predictions = model(srcs, src_lengths, tags, tgts, tgt_lengths)
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
        loss = compute_loss(predictions[:, :last_index -1], tgts[:, 1:last_index], sum(tgt_lengths) - batch_size) # TODO: make sure this is ok; bug prone line...

        total_loss += loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        nb += 1

    return total_loss / nb

def run_val_epoch(val_dataloader, model):
    total_loss = 0
    nb = 0
    for srcs, tgts, tags, src_lengths, tgt_lengths in val_dataloader:
        predictions = model(srcs, src_lengths, tags, tgts, tgt_lengths)
        last_index = max(tgt_lengths) # - 1 cause we don't predict BOS.

        # loss = criterion(predictions[:, last_index -1], tgts[:, 1:last_index]) # TODO: make sure this is ok; bug prone line...
        loss = compute_loss(predictions[:, :last_index -1], tgts[:, 1:last_index], sum(tgt_lengths) - batch_size) # TODO: make sure this is ok; bug prone line...

        total_loss += loss.data
        nb +=1
    return total_loss / nb


for i in range(5):
    model.train()
    epoch_loss = run_train_epoch(train_dataloader, model)
    print(f"Current loss: {epoch_loss}")

    model.eval()
    with torch.no_grad():
        val_loss = run_val_epoch(val_dataloader, model)
        print(f"Current validation loss: {val_loss}")
    