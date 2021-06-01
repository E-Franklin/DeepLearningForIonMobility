import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import onehot_encoding


# Recurrent neural network (many-to-one)
class RTLSTMOnehot(nn.Module):
    def __init__(self, input_size, num_lstm_units, num_layers, batch_size, vocab,
                 embed_dim=20, output_size=1):
        super(RTLSTMOnehot, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.num_lstm_units = num_lstm_units
        self.num_layers = num_layers

        # TODO: Test bidirectional LSTM
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_lstm_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=num_lstm_units, out_features=output_size)

    def forward(self, x, x_lengths):
        # x will have padded sequences
        x = onehot_encoding(x, self.embed_dim)

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        lstm_outs, (h_t, h_c) = self.lstm(x_packed)

        # Decode the hidden state of the last time step
        # TODO: test sigmoid vs ReLU
        out = self.fc(F.relu(h_t[-1, :, :]))
        return out
