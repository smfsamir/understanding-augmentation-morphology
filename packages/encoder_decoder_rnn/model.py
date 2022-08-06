import torch
import torch.nn as nn
from torch.nn import LSTM

from packages.transformer.self_attention_encoder import attention

class TwoStepDecoderCell(nn.Module):

    def __init__(self, char_embed: nn.Embedding, hidden_dim) -> None:
        """_summary_

        Args:
            char_embed (nn.Embedding): Embedding for the ground truth target sequence...? Make
            sure to use the same one for the recurrent encoder.
        """
        super().__init__()
        self.embedding = char_embed
        self.tag_attn = GlobalAttention()
        self.lemma_attn = GlobalAttention()
        self.rnn_cell = nn.GRUCell()
        self.attention_tags = GlobalAttention(hidden_dim)
        self.attention_lemma = GlobalAttention(hidden_dim)

    def forward(self, x, h_prev, annotations_tag, annotations_lemma, target_seq = None):
        """

        Args:
            x (Input token indices across a batch for a single time step): B x 1
            h_prev (torch.tensor): The hidden states from the previous time step. (B x D)
            annotations_tag (torch.tensor): B x S x D 
            annotations_lemma (torch.tensor): B x S x D.


        Returns:
            TODO
        """
        embed = self.embedding(x) # B x 1 x hidden_size
        embed = embed.squeeze(1) # B x hidden_size

        attention_weights_tag = self.attention_tags(h_prev, annotations_tag)
        tag_context_vec = (annotations_tag * attention_weights_tag).sum(axis=1)

        tag_enhanced_state = h_prev + tag_context_vec
        attention_weights_lemma = self.attention_lemmas(tag_enhanced_state, annotations_lemma)

        lemma_context_vec = (annotations_lemma * attention_weights_lemma).sum(axis=1)
        embed_and_context = torch.cat((lemma_context_vec, embed), dim=1)
        h_new = self.rnn(embed_and_context, h_prev)
        output = self.out(h_new)
        return output, h_new, attention_weights_lemma

class GlobalAttention(nn.Module):

    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.linear_cell_state = nn.Linear(hidden_dim, hidden_dim )
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.linear_input_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.global_attn_vector = nn.Linear(2* hidden_dim, 1)
    
    # NOTE: slightly inefficient, since we're computing the same annotation transformation every time. 
    def forward(self, decoder_hidden_states, annotations, mask):
        """

        Args:
            decoder_hidden_states (torch.tensor): The current decoder hidden state. B x D.
            annotations (torch.tensor): The encoder hidden states. B x S x D.
            mask (torch.tensor): Corresponds to padded elements of the . # TODO: must check whether to set this to true or false...

        Returns:
            torch.tensor: BxS Normalized attention weights for each source character in the batch. 
        """
        batch_size, seq_len, hid_size = annotations.size()
        expanded_hidden = decoder_hidden_states.unsqueeze(1).expand_as(annotations) # great.
        concat = torch.cat((expanded_hidden, annotations), dim=2)


        concat = torch.cat((expanded_hidden, annotations), dim=2) 
        reshaped_for_attention_net = concat.view((batch_size * seq_len, 2 * hid_size))
        attention_net_output = self.attention_network(reshaped_for_attention_net)
        unnormalized_attention = attention_net_output.view((batch_size, seq_len, 1)) # Reshape attention net output to have dimension batch_size x seq_len x 1
        unnormalized_attention[torch.bitwise_not(mask)] = float("-inf")
        return self.softmax(unnormalized_attention) # tested: see test_tensors.py:test_normalization_of_attention

class BidirectionalLemmaEncoder(nn.Module):

    def __init__(self, embedding_layer, hidden_size) -> None:
        self.embedding_layer = embedding_layer
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        hidden = self.init_hidden(batch_size)

        encoded = self.embedding(x)  # batch_size x seq_len x hidden_size
        annotations = []

        # NOTE: this is the only point where we could do anything the encoder ignore the padding somehow.
            # do we really need to do anything here...? The padding embedding is fixed. and it's 0
        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru(x, hidden)
            annotations.append(hidden)
        annotations = torch.stack(annotations, dim=1)
        return annotations, hidden