'''
Models for adversarial fine-tuning.
'''
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


def get_seq_len(src, batch_first):
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]
    
class BinaryClassificationHead(nn.Module):
    """Refer to Roberta classification head, since we are not using the pooling layer from Bert

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(input_size, 2)

    def forward(self, features):  
        x = self.dropout(features) # should be hidden state after mean pooling
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class AdversarialDecoder(nn.Module):
    """Transformer decoder adversary

    Args:
        nn (_type_): _description_
    """
    # torch 2.1 has bias in decoder layer and layer norm
    # torch 2.1 has tgt_is_causal
    def __init__(self, d_model=768, nhead=12, num_decoder_layers=2, dim_feedforward=256,
                 tgt_vocab_size=10000,
                 dropout=0.1, activation=F.relu, layer_norm_eps=1e-5,
                 batch_first=True, norm_first=False, device=None, dtype=None):
        super(AdversarialDecoder, self).__init__()
        # All you need from Bert is the vocab_size
        self.device = device
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, 
                                                   activation, layer_norm_eps, batch_first, 
                                                   norm_first)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory):
        # The input tgt should be Bert embedding from the lowest layer
        mask_1 = nn.Transformer.generate_square_subsequent_mask(get_seq_len(tgt, batch_first=True)).to(self.device)
        # mask_2 = nn.Transformer.generate_square_subsequent_mask(get_seq_len(memory, batch_first=True)).to(self.device)
        output = self.decoder(tgt, memory, tgt_mask=mask_1)
        # Map to vocab size
        output = self.generator(output)
        return output


