import torch.nn as nn
from torch.nn import LSTM

def make_model(char_vocab, tag_vocab, d_model=512, dropout=0.1):

    model = 


class TwoStepAttentionModel(nn.Module):
    def __init__(self, char_vocab_size: int, hidden_size: int):
        self.char_embeddings = 
    
    def forward(self, tag_sequence, input_sequence):
        h_tag = # TODO
        h_seq = # TODO: need to use a GRU cell here to collect the inputs for decoder attention. 




class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        self.embedding = nn.Embedding()