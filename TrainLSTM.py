import math

import numpy as np

import DataUtils
from RTLSTM import *
from SeqToIntTransform import *
from SequenceDataset import *


# will train on GPU CUDA cores if they are available in the system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 40
input_size = 20
num_lstm_units = 20
num_layers = 2
output_size = 1
batch_size = 4
num_epochs = 2
learning_rate = 0.00001

# define the possible characters in the sequence, - is used for padding
aas = '-ACDEFGHIKLMNPQRSTVWY'

# define the mappings for char to int and int to char
vocab = dict((a, i) for i, a in enumerate(aas))
to_int = SeqToInt(vocab)
rt_data = SequenceDataset('data\\dia.txt', 'RT', transform=to_int)

# plot the targets before scaling
targets = rt_data.get_targets()
DataUtils.get_stats(targets)


# scale the data and plot again. Scaling doesn't help.
# rt_data.scale_targets()
# targets = rt_data.get_targets()
# DataUtils.get_stats(targets)

num_data_points = len(rt_data)

train_split = math.floor(0.7 * num_data_points)
test_split = math.floor(0.2 * num_data_points)
val_split = num_data_points - train_split - test_split

train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(rt_data,
                                                                           [train_split,
                                                                            test_split,
                                                                            val_split
                                                                            ])


def pad_sort_collate(batch):
    """
    :param batch: A batch of samples of the form seq:target:length
    :return: the padded batch and lengths of the sequences
    """
    # get the sequence and targets as lists
    # TODO: find a better way to do this
    batch_seqs, batch_targets, lens = [s['sequence'] for s in batch], \
                                      [s['target'] for s in batch], [s['length'] for s in batch]

    # pad the sequences
    max_length = max(lens)
    padded_seqs = np.zeros((len(batch), max_length), dtype=int)

    for i, seq_length in enumerate(lens):
        padded_seqs[i, 0:seq_length] = batch_seqs[i][0:seq_length]

    # convert to tensors
    padded_seqs = torch.tensor(padded_seqs)
    batch_targets = torch.tensor(batch_targets)
    batch_lengths = torch.tensor(lens)

    # sort the sequences in the batch from longest to shortest
    batch_lengths, perm_idx = batch_lengths.sort(0, descending=True)
    padded_seqs = padded_seqs[perm_idx]
    batch_targets = batch_targets[perm_idx]

    return padded_seqs, batch_targets, batch_lengths


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler=None,
                                           batch_sampler=None,
                                           collate_fn=pad_sort_collate,
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)

model = RTLSTM(input_size, num_lstm_units, num_layers, batch_size, vocab, device).to(device)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (seqs, targets, lengths) in enumerate(train_loader):
        seqs = seqs.to(device).long()
        targets = targets.view(batch_size, 1).to(device).float()

        # Forward pass
        outputs = model(seqs, lengths)
        loss = loss_fn(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            losses.append(loss.item())
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

DataUtils.get_stats(losses)
