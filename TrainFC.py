import math

import numpy as np
import torch
import pandas as pd
import torch.nn as nn

import DataUtils
from FullyConnectedModel import FCModel
from SeqToIntTransform import SeqToInt
from SequenceDataset import SequenceDataset


def pad_collate(batch):
    """
    :param batch: A batch of samples of the form seq:target:length
    :return: the padded batch and lengths of the sequences
    """
    # get the sequence and targets as lists
    # TODO: find a better way to do this
    batch_seqs, batch_targets, lens = [s['sequence'] for s in batch], \
                                      [s['target'] for s in batch], [s['length'] for s in batch]

    # pad the sequences
    max_length = input_size
    padded_seqs = np.zeros((len(batch), max_length))

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

    return padded_seqs, batch_targets


# will train on GPU CUDA cores if they are available in the system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 40
batch_size = 4
num_epochs = 4
learning_rate = 0.00001

# define the possible characters in the sequence, - is used for padding
aas = '-ACDEFGHIKLMNPQRSTVWY'

# define the mappings for char to int and int to char
vocab = dict((a, i) for i, a in enumerate(aas))
to_int = SeqToInt(vocab)
data_frame = pd.read_csv('data\\dia.txt', sep=None)[['sequence', 'RT']]

# remove all sequences longer than 40
data_frame = data_frame[data_frame['sequence'].str.len() <= input_size]
data_frame.reset_index(drop=True, inplace=True)

rt_data = SequenceDataset(data_frame, 'RT', transform=to_int)
num_data_points = len(rt_data)

train_split = math.floor(0.7 * num_data_points)
test_split = math.floor(0.2 * num_data_points)
val_split = num_data_points - train_split - test_split

train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(rt_data,
                                                                           [train_split,
                                                                            test_split,
                                                                            val_split
                                                                            ])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler=None,
                                           batch_sampler=None,
                                           collate_fn=pad_collate,
                                           shuffle=True,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=True)

model = FCModel(input_size, batch_size, device).to(device)

# Loss and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
sum_loss = 0
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (seqs, targets) in enumerate(train_loader):
        seqs = seqs.to(device).float()
        targets = targets.view(batch_size, 1).to(device).float()

        # Forward pass
        outputs = model(seqs)
        loss = loss_fn(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        # clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()
        sum_loss += loss.item()
        if (i + 1) % 100 == 0:
            losses.append(sum_loss/100)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, sum_loss/100))
            print(pd.DataFrame({'Target': list(targets.data.cpu().numpy()),
                                'Output': list(outputs.data.cpu().numpy())}, columns=['Target', 'Output']))
            sum_loss = 0

DataUtils.plot_losses(losses)
