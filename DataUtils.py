import re
from statistics import median

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import wandb
from SeqToIntTransform import *
from SequenceDataset import *




def delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]


def delta_tr95(act, pred):
    return delta_t95(act, pred) / (max(act) - min(act))


def med_rel_error(act, pred):
    return median(abs(act - pred) / act) * 100


def delta_t90(act, pred):
    num90 = int(np.ceil(len(act) * 0.90))
    return 2 * sorted(abs(act - pred))[num90 - 1]


def delta_t90_err(act, pred):
    err = abs(act - pred) / act
    return delta_t90(err, 0)


def get_stats(data):
    if type(data) is list:
        data = pd.Series(data)
    # plot_series(data)
    mean = data.mean()
    stdev = data.std()
    print(f'Mean: {mean:.3f} stdev: {stdev:.3f}')


def get_vocab():
    # define the possible characters in the sequence, - is used for padding. a is acetylation and m is methionine oxidation.
    # This will be the same for all datasets as they will be preprocessed. The size of this list affects the encoding.
    aas = 'ACDEFGHIKLMNPQRSTVWYamc'
    aa_charge = ['-']

    if wandb.config.use_charge:
        # for every character encoding of amino acids that we have defined, apend the charge and add it to the list.
        # charges range from 1 to 5
        # TODO: build the vocabulary from the data
        for a in aas:
            for i in range(1, 6):
                if a + str(i) not in aa_charge:
                    aa_charge.append(a + str(i))
        vocab = dict((a, i) for i, a in enumerate(aa_charge))
    else:
        # define the mappings for char to int and int to char
        aas += aa_charge[0]
        vocab = dict((a, i) for i, a in enumerate(aas))

    # wandb.config.embedding_dim = len(vocab)
    return vocab


def load_file(filename):
    if wandb.config.use_charge:
        data_frame = pd.read_csv(filename, sep='\t')[
            ['sequence', 'charge', wandb.config.target]]
        data_set = SequenceChargeDataset(data_frame, wandb.config.target)
                                        # transform=ChargeSeqToInt(get_vocab()))
    else:
        data_frame = pd.read_csv(filename, sep='\t')[
            ['sequence', wandb.config.target]]
        data_set = SequenceDataset(data_frame, wandb.config.target,
                                   transform=SeqToInt(get_vocab()))

    return data_set


def load_training_data(collate_fn):
    file = wandb.config.data_dir + wandb.config.data_set + '_train.tsv'
    data_set = load_file(file)
    wandb.config.max_length = data_set.get_max_length()

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set.scale_targets(scaler)

    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=wandb.config.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               drop_last=True)
    return train_loader, scaler


def load_testing_data(collate_fn, scaler):
    file = wandb.config.data_dir + wandb.config.data_set + '_test.tsv'
    data_set = load_file(file)
    print(data_set.get_max_length())
    data_set.scale_targets(scaler)
    if wandb.config.max_length < data_set.get_max_length():
        wandb.config.update({'max_length': data_set.get_max_length()},
                            allow_val_change=True)

    test_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=wandb.config.batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=True)
    return test_loader


def load_validation_data(collate_fn, scaler):
    file = wandb.config.data_dir + wandb.config.data_set + '_val.tsv'
    data_set = load_file(file)
    print(data_set.get_max_length())
    data_set.scale_targets(scaler)
    if wandb.config.max_length < data_set.get_max_length():
        wandb.config.update({'max_length': data_set.get_max_length()},
                            allow_val_change=True)

    val_loader = torch.utils.data.DataLoader(dataset=data_set,
                                             batch_size=wandb.config.batch_size,
                                             collate_fn=collate_fn,
                                             shuffle=False,
                                             drop_last=True)
    return val_loader
