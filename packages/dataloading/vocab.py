import torch
from typing import Iterable, Iterator, Tuple, List
from torchtext.vocab import build_vocab_from_iterator, Vocab
import pandas as pd
from packages.pkl_operations.pkl_io import load_pkl_from_path 
from os.path import exists

def yield_tokens_char(data_iter: Iterator[Tuple[int, pd.Series]]) -> Iterable[List[str]]:
    for _, src_tgt_row in data_iter:
        yield list(src_tgt_row['src']) + list(src_tgt_row['tgt'])

def yield_tokens_tag(data_iter: Iterator[str]) -> Iterable[List[str]]:
    for _, tag_row in data_iter:
        yield (tag_row['tag'].split(';')) 

def build_vocabulary(sigm_df: pd.DataFrame) -> Vocab :
    vocab_char = build_vocab_from_iterator(
        yield_tokens_char(sigm_df[['src', 'tgt']].iterrows()),
        specials=["<s>", "</s>", "<unk>", "<blank>"]
    )

    vocab_tag = build_vocab_from_iterator(
        yield_tokens_tag(sigm_df[['tag']].iterrows()),
        specials=["<s>", "</s>", "<unk>", "<blank>"]
    )
    return (vocab_char, vocab_tag)

def load_vocab(sigm_df: pd.DataFrame) -> Tuple[Vocab, Vocab]:
    if not exists("data/vocab.pt"):
        char_vocab, tag_vocab  = build_vocabulary(sigm_df)
        # torch.save((char_vocab, tag_vocab), "data/vocab.pt")
    else:
        char_vocab, tag_vocab = torch.load("data/vocab.pt")
    return (char_vocab, tag_vocab)