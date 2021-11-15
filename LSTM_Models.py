import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import onehot_encoding
from path_config import cpcprot_dir
from CPCProt import CPCProtModel, CPCProtEmbedding


# Recurrent neural network (many-to-one)
class LSTMOnehot(nn.Module):
    def __init__(self, embedding_dim, num_lstm_units, num_layers, vocab, bidirectional, use_charge, dropout=0, output_size=1):
        super(LSTMOnehot, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.num_lstm_units = num_lstm_units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_charge = use_charge

        ckpt_path = cpcprot_dir + "data/best.ckpt"  # Replace with actual path to CPCProt weights
        model = CPCProtModel()
        model.load_state_dict(torch.load(ckpt_path))
        self.embedder = CPCProtEmbedding(model)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=num_lstm_units, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

        linear_input = num_lstm_units
        if bidirectional:
            linear_input = linear_input*2
        if use_charge:
            linear_input = linear_input+1

        self.fc1 = nn.Linear(in_features=linear_input, out_features=num_lstm_units)
        self.fc2 = nn.Linear(in_features=num_lstm_units, out_features=output_size)

    def forward(self, x, charges, x_lengths):
        # x will have padded sequences
        #x = onehot_encoding(x, self.embedding_dim)
        # $z$ is the output of the CPCProt encoder
        z = [self.embedder.get_z(s) for s in x]

        z = torch.stack(z)
        # remove the extra dim from the CPCProt embedder (it is the batch dim
        # from the embedder but the actual batch size is added in torch.stack)
        z = torch.squeeze(z, dim=1)

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # x_packed = torch.nn.utils.rnn.pack_padded_sequence(z, x_lengths, batch_first=True)

        outputs, (h_t, h_c) = self.lstm(z)

        # Decode the hidden state of the last time step
        if self.bidirectional:
            lstm_out = torch.cat((h_t[-1, :, :], h_t[-2, :, :]), dim=1)
        else:
            lstm_out = h_t[-1, :, :]

        if self.use_charge:
            lstm_out = torch.cat([lstm_out, charges], dim=1)

        o1 = F.relu(self.fc1(lstm_out))
        out = self.fc2(o1)
        return out
