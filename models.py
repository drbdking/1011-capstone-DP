'''
Models for adversarial fine-tuning.
'''
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch import functional as F
from transformers import BertModel


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
    ):
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )

def _get_seq_len(src, batch_first):
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
        
def _detect_is_causal_mask(mask, is_causal=None, size=None,):
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

    
class BinaryClassificationHead(nn.Module):
    """Refer to Roberta classification head, since we are not using the pooling layer from Bert

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size) -> None:
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.5)
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
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 tgt_vocab_size=10000, 
                 dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, 
                 batch_first=False, norm_first=False, bias=True, device=None, dtype=None) -> None:
        super().__init__()
        # All you need from Bert is the vocab_size
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, 
                                                    norm_first, bias)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def foward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # The input tgt should be Bert embedding from the lowest layer
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_is_causal=tgt_is_causal,)

        if self.norm is not None:
            output = self.norm(output)

        # Map to vocab size
        output = self.generator(output)
        return output
    

