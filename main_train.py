from packages.dataloading import vocab
from packages.dataloading.morphology_dataloader import create_dataloader
from packages.dataloading.morphology_dataset import SigmorphonDataset
from packages.dataloading.vocab import load_vocab
from packages.utils.util_functions import get_mask

from packages.transformer.self_attention_encoder import make_model
from packages.encoder_decoder_rnn.model import BidirectionalLemmaEncoder, TwoStepDecoderCell

import pandas as pd
import torch.nn as nn
import torch.optim as optim

# NOTE: this should work for just morphological inflection. Reinflection is a bit more delicate.
sigm_df = pd.read_csv("data/eng_1000_train.tsv", sep='\t')
vocab_char, vocab_tag = load_vocab(sigm_df)
padding_id = vocab_char.get_stoi()["<blank>"]

# TODO: make train/validation split
sigm_dataset = SigmorphonDataset(sigm_df)
dataloader = create_dataloader(sigm_dataset, 32, vocab_char, vocab_tag, 'cpu')

hidden_dim = 64
embed_layer = nn.Embedding(len(vocab_char) + 1, embedding_dim=hidden_dim, padding_idx=padding_id) # +1 for padding token?
tag_encoder = make_model(len(vocab_tag), d_model=hidden_dim)
lemma_encoder = BidirectionalLemmaEncoder(embed_layer, hidden_dim)
decoder = TwoStepDecoderCell(embed_layer, hidden_dim, len(vocab_char))

criterion = nn.CrossEntropyLoss(ignore_index=padding_id, reduction='sum')
optimizer = optim.Adam(list(lemma_encoder.parameters()) + list(decoder.parameters()) 
                       + list(tag_encoder.parameters()), lr=0.01)

# TODO: make train and validation loop.
# TODO: we should also try generating an example.

def run_train_epoch(train_dataloader):
    for srcs, tgts, tags, src_lengths, tgt_lengths in dataloader:
    # dataloader_iter = iter(dataloader)
    # srcs, tgts, tags = next(dataloader_iter)
        annotations_tag = tag_encoder(tags, None) 
        annotations_lemma, src_embeds_final = lemma_encoder(srcs, src_lengths) 

        attn_mask = get_mask(srcs, padding_id) 

        decoder_hidden = src_embeds_final
        tgt_seq_len = tgts.shape[1] -1 # -1 because we start with the bos token; we don't compute loss on it

        decoder_input = tgts[:,0]
        loss = 0
        for i in range(1, tgt_seq_len): # start from 1 because we started with the bos token.
            decoder_output, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, annotations_tag, annotations_lemma, attn_mask)

            current_target = tgts[:, i]
            if all(current_target == padding_id): 
                continue
            loss += criterion(decoder_output, current_target)
            decoder_input = tgts[:,i]     
        loss /= sum(tgt_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.data


# # print(srcs)
# print(srcs)

