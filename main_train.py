
from packages.dataloading import vocab
from packages.dataloading.morphology_dataloader import create_dataloader
from packages.dataloading.morphology_dataset import SigmorphonDataset
from packages.dataloading.vocab import load_vocab

import pandas as pd

from packages.transformer.self_attention_encoder import make_model

sigm_df = pd.read_csv("data/eng_1000_train.tsv", sep='\t')
print(sigm_df.columns)
vocab_char, vocab_tag = load_vocab(sigm_df)
print(vocab_char)

sigm_dataset = SigmorphonDataset(sigm_df)

dataloader = create_dataloader(sigm_dataset, 32, vocab_char, vocab_tag, 'cpu')
dataloader_iter = iter(dataloader)

srcs, tgts, tags = next(dataloader_iter)

tag_encoder = make_model(len(vocab_tag))
tag_encoder(tags, None)

def run_epoch():
    
# print(srcs)
print(srcs)


