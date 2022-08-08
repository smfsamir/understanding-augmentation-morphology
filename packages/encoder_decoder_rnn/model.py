import torch
import torch.nn as nn
from torch.nn import LSTM
from packages.utils.util_functions import get_mask

from ..transformer.self_attention_encoder import make_model

class TwoStepAttentionInflector(nn.Module):

    def __init__(self, lemma_encoder, tag_encoder, two_step_decoder, padding_id) -> None:
        super().__init__()
        self.lemma_encoder = lemma_encoder
        self.tag_encoder = tag_encoder
        self.two_step_decoder = two_step_decoder
        self.padding_id = padding_id
    
    def forward(self, src, src_lengths, tags, tgts = None, tgt_lengths = None):
        """_summary_

        Args:
            use_tf (bool): Whether or not to use teacher-forcing (training).

        Returns:
            _type_: _description_
        """
        annotations_tag = self.tag_encoder(tags, None) 
        annotations_lemma, src_embeds_final = self.lemma_encoder(src, src_lengths) 

        attn_mask = get_mask(src, self.padding_id) 

        decoder_input = tgts[:,0]
        decoder_hidden = src_embeds_final
        tgt_seq_len = tgts.shape[1] -1 # -1 because we start with the bos token; we don't compute loss on it

        decoder_input = tgts[:,0]
        decoder_outputs = []
        for i in range(1, tgt_seq_len): # start from 1 because we started with the bos token.
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, annotations_tag, annotations_lemma, attn_mask)

            decoder_outputs.append(decoder_output)
            decoder_hidden = src_embeds_final
            decoder_input = tgts[:,i]     
        return torch.cat(decoder_outputs, dim=1) 

class TwoStepDecoderCell(nn.Module):

    def __init__(self, char_embed: nn.Embedding, hidden_dim: int, vocab_size: int) -> None:
        """_summary_

        Args:
            char_embed (nn.Embedding): Embedding for the ground truth target sequence...? Make
            sure to use the same one for the recurrent encoder.
        """
        super().__init__()
        self.embedding = char_embed
        self.tag_attn = GlobalAttention(hidden_dim)
        self.lemma_attn = GlobalAttention(hidden_dim)
        self.rnn_cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.attention_tags = GlobalAttention(hidden_dim)
        self.attention_lemma = GlobalAttention(hidden_dim)
        # NOTE: vocab size is not quite right i think, since we have those 
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h_prev, annotations_tag, annotations_lemma, lemma_mask):
        """
        Args:
            x (Input token indices across a batch for a single time step): B x 1
            h_prev (torch.tensor): The hidden states from the previous time step. (B x D)
            annotations_tag (torch.tensor): B x S x D 
            annotations_lemma (torch.tensor): B x S x D.

        Returns:
        """
        embed = self.embedding(x) # B x 1 x hidden_size
        embed = embed.squeeze(1) # B x hidden_size

        tag_mask = None
        attention_weights_tag = self.attention_tags(h_prev, annotations_tag, tag_mask)
        tag_context_vec = (annotations_tag * attention_weights_tag).sum(axis=1)

        tag_enhanced_state = h_prev + tag_context_vec
        attention_weights_lemma = self.attention_lemma(tag_enhanced_state, annotations_lemma, lemma_mask)

        lemma_context_vec = (annotations_lemma * attention_weights_lemma).sum(axis=1)
        embed_and_context = torch.cat((lemma_context_vec, embed), dim=1)
        h_new = self.rnn_cell(embed_and_context, h_prev)
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
        self.softmax = nn.Softmax(dim=1)
    
    # NOTE: slightly inefficient, since we're computing the same annotation transformation every time. 
    def forward(self, decoder_hidden_states, annotations, mask=None):
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

        reshaped_for_attention_net = concat.view((batch_size * seq_len, 2 * hid_size))
        attention_net_output = self.attention_network(reshaped_for_attention_net)
        unnormalized_attention = attention_net_output.view((batch_size, seq_len, 1)) # Reshape attention net output to have dimension batch_size x seq_len x 1
        if mask is not None:
            unnormalized_attention[(mask)] = float("-inf")
        return self.softmax(unnormalized_attention) # tested: see test_tensors.py:test_normalization_of_attention

class BidirectionalLemmaEncoder(nn.Module):

    def __init__(self, embedding_layer, hidden_size) -> None:
        super(BidirectionalLemmaEncoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.hidden_size = hidden_size
    
    def _init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)

    def forward(self, x, src_lengths):
        """

        Args:
            x (): Token indices. BS x seq_len
            src_lengths (_type_): BS  

        Returns:
            : (BS x seq_len x D), (BS x D)
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        hidden = self._init_hidden(batch_size)

        encoded = self.embedding_layer(x)  # batch_size x seq_len x hidden_size
        annotations = []

        # NOTE: this is the only point where we could do anything the encoder ignore the padding somehow.
            # do we really need to do anything here...? The padding embedding is fixed. and it's 0
        for i in range(seq_len):
            x = encoded[:,i,:]  # Get the current time step, across the whole batch
            hidden = self.gru_cell(x, hidden)
            annotations.append(hidden)
        annotations = torch.stack(annotations, dim=1)
        final_hidden = annotations[torch.arange(32), src_lengths]
        return annotations, final_hidden
    
def make_inflector(vocab_char, vocab_tag, padding_id, hidden_dim=64):
    hidden_dim = 64
    embed_layer = nn.Embedding(len(vocab_char) + 1, embedding_dim=hidden_dim, padding_idx=padding_id) # +1 for padding token?
    tag_encoder = make_model(len(vocab_tag), d_model=hidden_dim)
    lemma_encoder = BidirectionalLemmaEncoder(embed_layer, hidden_dim)
    decoder = TwoStepDecoderCell(embed_layer, hidden_dim, len(vocab_char))

    return TwoStepAttentionInflector(lemma_encoder, tag_encoder, decoder)
