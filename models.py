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
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
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
                 batch_first=True, norm_first=False, use_separate_embedding=False, device=None, dtype=None):
        super(AdversarialDecoder, self).__init__()
        # All you need from Bert is the vocab_size
        self.device = device
        self.use_separate_embedding = use_separate_embedding
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, 
                                                   activation, layer_norm_eps, batch_first, 
                                                   norm_first)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory):
        if self.use_separate_embedding:
            # tgt is the Tensor that stores the index of words
            print(1)
            tgt = self.embedding(tgt)
        # The input tgt should be Bert embedding from the lowest layer or the output from embedding layer of this model
        mask_1 = nn.Transformer.generate_square_subsequent_mask(get_seq_len(tgt, batch_first=True)).to(self.device)
        # mask_2 = nn.Transformer.generate_square_subsequent_mask(get_seq_len(memory, batch_first=True)).to(self.device)
        output = self.decoder(tgt, memory, tgt_mask=mask_1)
        # Map to vocab size
        output = self.generator(output)
        return output


class MultiSetInversionModel(nn.Module):
    def __init__(self, emb_dim, output_size, steps=32, drop_p=0.25, device="cpu"):
        super(MultiSetInversionModel, self).__init__()
        self.device = device
        self.steps = steps
        self.emb_dim = emb_dim
        self.output_size = output_size
        self.device = device
   
        # Initialize layers and move them to the GPU if available
        self.fc1 = nn.Linear(emb_dim, emb_dim)  # on input
        self.fc2 = nn.Linear(emb_dim, output_size)  # on output of lstm cell
        self.policy = nn.LSTMCell(emb_dim, emb_dim)
        self.dropout = nn.Dropout(drop_p)

        # Initialize embedding  
        self.embedding = nn.Embedding(output_size, emb_dim)

    def forward(self, inputs, labels):
        # Labels should be bool
        xt = self.fc1(inputs)
        states=None
        batch_size = labels.size(0)
        init_labels_t = labels.clone()
        init_input_t = xt
        init_states_t = states
        init_prediction_t = torch.zeros_like(labels, dtype=torch.bool, device=self.device)  
        init_loss_t = torch.zeros_like(labels, dtype=torch.float64, device=self.device)

        i = 0
        while i < self.steps:
            i, init_labels_t, init_input_t, init_states_t, init_prediction_t, init_loss_t = \
                self._inner_loop(i, init_labels_t, init_input_t, init_states_t, init_prediction_t, init_loss_t)
            
        final_loss = torch.mean(init_loss_t / self.steps)

        return init_prediction_t, final_loss
    
    def _inner_loop(self, i, labels_t, input_t, states_t, prediction_t, loss_t):
        input_t = self.dropout(input_t)
        states_t = self.policy(input_t, states_t)
        logits = self.fc2(states_t[0])

        yt = torch.argmax(logits, dim=1)

        # Get embedding and record all predicted words
        input_t = self.embedding(yt) 
        yt_one_hot = F.one_hot(yt, num_classes=self.output_size).to(torch.bool)
        prediction_t = torch.logical_or(yt_one_hot, prediction_t)

        # A record of word remained unpredicted
        labels_t = torch.logical_and(labels_t, torch.logical_not(yt_one_hot))
        # Did nothing here
        mask = labels_t.clone().to(torch.float32)


        logits_train = logits
        # Mask
        loss = -F.log_softmax(logits_train, dim=1) * mask
        loss_t += loss

        return i + 1, labels_t, input_t, states_t, prediction_t, loss_t

