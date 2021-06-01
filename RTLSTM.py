import torch
import torch.nn as nn
import torch.nn.functional as F


# Recurrent neural network (many-to-one)
class RTLSTM(nn.Module):
    def __init__(self, input_size, num_lstm_units, num_layers, batch_size, vocab,
                 embed_dim=20, output_size=1):
        super(RTLSTM, self).__init__()
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.num_lstm_units = num_lstm_units
        self.num_layers = num_layers

        # this embedding will make the vector all 0 for any padding character
        self.word_embedding = nn.Embedding(num_embeddings=(len(self.vocab)), embedding_dim=self.embed_dim,
                                           padding_idx=self.vocab['-'])
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_lstm_units, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(num_lstm_units, output_size)

    def forward(self, x, x_lengths):
        # x will have padded sequences

        x = self.word_embedding(x)

        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, num_lstm_units)
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        lstm_outs, (h_t, h_c) = self.lstm(x_packed)

        # undo the packing operation
        # lstm_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_outs, batch_first=True)

        # Decode the hidden state of the last time step
        out = self.fc(F.relu(h_t[-1, :, :]))

        return out
