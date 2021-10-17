import math

import torch
from CPCProt import CPCProtModel, CPCProtEmbedding

from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from path_config import cpcprot_dir


class SequenceTransformer(nn.Module):
    def __init__(self, vocab, embed_size, nhead, dim_ff, num_layers, use_charge,
                 dropout=0):
        super(SequenceTransformer, self).__init__()

        self.use_charge = use_charge
        # Load pretrained model to use for embedding
        ckpt_path = cpcprot_dir + "data/best.ckpt"  # Replace with actual path to CPCProt weights
        model = CPCProtModel()
        model.load_state_dict(torch.load(ckpt_path))
        self.embedder = CPCProtEmbedding(model)

        # self.encoder = nn.Embedding(num_embeddings=(len(vocab)), embedding_dim=embed_size,
        #                            padding_idx=vocab['-'])

        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=embed_size,
                                                 nhead=nhead,
                                                 dim_feedforward=dim_ff,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      num_layers)
        self.d_model = embed_size

        if use_charge:
            self.decoder1 = nn.Linear(embed_size + 1, 60)
        else:
            self.decoder1 = nn.Linear(embed_size, 60)

        self.decoder2 = nn.Linear(60, 1)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, seq, charges):
        # seq = self.encoder(seq) * math.sqrt(self.d_model)

        z = []
        for s in seq:
            # $z$ is the output of the CPCProt encoder
            z.append(self.embedder.get_z(s))  # (1, L // 11, 512)

        z = torch.stack(z)
        # remove the extra dim from the CPCProt embedder
        z = torch.squeeze(z, dim=1)
        seq = self.pos_encoder(z)

        t_output = self.transformer_encoder(seq)
        # t_output = self.transformer_encoder(seq, src_mask)

        t_output = torch.sum(t_output, dim=1)
        if self.use_charge:
            t_output = torch.cat([t_output, charges], dim=1)

        output = self.decoder1(t_output)
        output = self.decoder2(output)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
