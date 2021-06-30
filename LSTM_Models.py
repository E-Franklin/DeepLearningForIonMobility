import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import onehot_encoding


# Recurrent neural network (many-to-one)
class LSTMOnehot(nn.Module):
    def __init__(self, embedding_dim, num_lstm_units, num_layers, vocab, bidirectional, use_charge, output_size=1):
        super(LSTMOnehot, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.num_lstm_units = num_lstm_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_charge = use_charge

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=num_lstm_units, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)

        self.fc_wcharge = nn.Linear(in_features=num_lstm_units+1, out_features=output_size)
        self.fc = nn.Linear(in_features=num_lstm_units, out_features=output_size)

        self.bidir_wcharge_fc = nn.Linear(in_features=2*num_lstm_units + 1, out_features=output_size)
        self.bidir_fc = nn.Linear(in_features=2*num_lstm_units, out_features=output_size)

    def forward(self, x, charges, x_lengths):
        # x will have padded sequences
        x = onehot_encoding(x, self.embedding_dim)

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        outputs, (h_t, h_c) = self.lstm(x_packed)

        # Decode the hidden state of the last time step
        if self.bidirectional:
            lstm_out = torch.cat((h_t[-1, :, :], h_t[-2, :, :]), dim=1)
            if self.use_charge:
                lstm_out = torch.cat([lstm_out, charges], dim=1)
                out = self.bidir_wcharge_fc(F.relu(lstm_out))
            else:
                out = self.bidir_fc(F.relu(lstm_out))
        else:
            lstm_out = h_t[-1, :, :]
            if self.use_charge:
                lstm_out = torch.cat([lstm_out, charges], dim=1)
                out = self.fc_wcharge(F.relu(lstm_out))
            else:
                out = self.fc(F.relu(lstm_out))

        return out
