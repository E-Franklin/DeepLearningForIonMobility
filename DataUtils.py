from statistics import median

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import wandb
from SeqToIntTransform import *
from SequenceDataset import *
import math
from pandas import DataFrame
from scipy import constants


def delta_t95(act, pred):
    """
    :param act: array containing the actual data values
    :param pred: array containing the predicted data values
    :return: the width of the interval containing 95% of the residual values
    """
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
    """
    :param act: array containing the actual data values
    :param pred: array containing the predicted data values
    :return: The width of the window containing 90% of the relative % error values
    """
    err = (abs(act - pred) / act) * 100
    return delta_t90(err, 0)


def get_stats(data):
    if type(data) is list:
        data = pd.Series(data)
    # plot_series(data)
    mean = data.mean()
    stdev = data.std()
    print(f'Mean: {mean:.3f} stdev: {stdev:.3f}')


def get_vocab():
    # define the possible characters in the sequence, - is used for padding. a
    # is acetylation and m is methionine oxidation. This will be the same for
    # all datasets as they will be preprocessed. The size of this list affects
    # the encoding.
    aas = 'ACDEFGHIKLMNPQRSTVWYamc'
    aa_charge = ['-']

    if wandb.config.use_charge:
        # for every character encoding of amino acids that we have defined,
        # append the charge and add it to the list. charges range from 1 to 5
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

    wandb.config.embedding_dim = len(vocab)
    return vocab


def load_file(filename):
    data_frame = pd.read_csv(filename, sep='\t')

    if wandb.config.use_charge:
        transform = ChargeSeqToInt(get_vocab())
    else:
        transform = SeqToInt(get_vocab())

    data_set = SequenceDataset(data_frame, wandb.config.target,
                               transform)

    return data_set


def load_training_data(collate_fn):
    file = wandb.config.training_data
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


def load_test_data(collate_fn, scaler):
    file = wandb.config.testing_data
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


def load_pred_data(collate_fn, data_file):
    data_frame = pd.read_csv(data_file)

    if wandb.config.use_charge:
        transform = ChargeSeqToInt(get_vocab())
    else:
        transform = SeqToInt(get_vocab())

    data_set = SequencePredDataset(data_frame, transform)
    print(data_set.get_max_length())

    if wandb.config.max_length < data_set.get_max_length():
        wandb.config.update({'max_length': data_set.get_max_length()},
                            allow_val_change=True)

    pred_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=wandb.config.batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=True)
    return data_frame, pred_loader


def k0_to_ccs(mz, charge, im):
    # constants
    n0 = constants.value(
        u'Loschmidt constant (273.15 K, 101.325 kPa)')  # Loschmidt's number
    kb = constants.value(u'Boltzmann constant')  # JK-1
    uamu = constants.value(u'unified atomic mass unit')  # 1/12 12C in kg
    # elementary charge C = A*s (Ampere*second)
    e = constants.value(u'elementary charge')
    t = 305  # K
    # t0 = 273.15  # K
    # p = 2.7  # mbar, 0.002664692820133235 atm
    # p0 = 1013.25  # mbar, 1 atm
    mg = 28  # mass of N2 in Da

    # calculate the CCS from 1/k0
    mi = mz * charge
    mu_kg = ((mi * mg) / (mi + mg)) * uamu  # convert Da to kg
    term2 = np.sqrt((2 * constants.pi) / (mu_kg * kb * t))

    k0 = 1 / im
    # 10 ** -4 convert cm**2 to m**2
    term3 = (charge * e) / ((k0 * 10 ** -4) * n0)

    ccs = (3 / 16) * term2 * term3 * 10 ** 20  # 10**20 converts m**2 to Angstrom
    print(ccs)
    # print(data.loc[i, 'CCS'])

    return ccs


def ccs_to_k0(mz, charge, ccs):
    # constants
    n0 = constants.value(
        u'Loschmidt constant (273.15 K, 101.325 kPa)')  # Loschmidt's number
    kb = constants.value(u'Boltzmann constant')  # J*K-1
    uamu = constants.value(u'unified atomic mass unit')  # 1/12 12C in kg
    # elementary charge C = A*s (Ampere*second)
    e = constants.value(u'elementary charge')
    t = 305  # K
    # t0 = 273.15  # K
    # p = 2.7  # mbar, 0.002664692820133235 atm
    # p0 = 1013.25  # mbar, 1 atm
    mg = 28  # mass of N2 in Da

    # calculate the CCS from 1/k0
    mi = mz * charge
    mu_kg = ((mi * mg) / (mi + mg)) * uamu  # convert Da to kg
    term2 = np.sqrt((2 * constants.pi) / (mu_kg * kb * t))

    # 10**20 converts Angstrom to m**2
    term3 = (charge * e) / ((ccs * 10 ** 20) * n0)

    k0 = (3 / 16) * term2 * term3 * 10 ** -4
    print(1/k0)
    # print(data.loc[i, 'CCS'])

    return 1/k0




