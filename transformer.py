import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SequenceTransformer(nn.Module):
    def __init__(self, vocab, embed_size, nhead, dim_ff, num_layers, dropout=0.5):
        super(SequenceTransformer, self).__init__()
        self.encoder = nn.Embedding(num_embeddings=(len(vocab)), embedding_dim=embed_size,
                                    padding_idx=vocab['-'])
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=dim_ff,
                                                 dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.d_model = embed_size
        self.decoder = nn.Linear(embed_size, 1)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        t_output = self.transformer_encoder(src, src_mask)
        t_output = torch.sum(t_output, dim=1)
        output = self.decoder(t_output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
