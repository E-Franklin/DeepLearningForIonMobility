import numpy as np
import seaborn as sns
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from SequenceDataset import *
from SeqToIntTransform import *
from sklearn.preprocessing import MinMaxScaler

from path_config import data_dir


def delta_t95(act, pred):
    num95 = int(np.ceil(len(act) * 0.95))
    return 2 * sorted(abs(act - pred))[num95 - 1]


def delta_tr95(act, pred):
    return delta_t95(act, pred) / (max(act) - min(act))


def plot_series(data, title=''):
    sns.displot(data, kind='kde')
    plt.subplots_adjust(top=.90)
    plt.title(title, y=1.04)
    plt.savefig('plots/' + title + '.png')
    # wandb.log({'title': plt})
    plt.show()


def get_stats(data):
    if type(data) is list:
        data = pd.Series(data)
    plot_series(data)
    mean = data.mean()
    stdev = data.std()
    print(f'Mean: {mean:.3f} stdev: {stdev:.3f}')


def get_vocab():
    # define the possible characters in the sequence, - is used for padding. a is acetylation and m is methionine oxidation.
    # This will be the same for all datasets as they will be preprocessed. The size of this list affects the encoding.
    aas = '-ACDEFGHIKLMNPQRSTVWYam'

    # define the mappings for char to int and int to char
    vocab = dict((a, i) for i, a in enumerate(aas))
    return vocab


def load_training_data(collate_fn):
    file = data_dir + wandb.config.data_set + '_train.tsv'
    data_frame = pd.read_csv(file, sep='\t')[['sequence', wandb.config.target]]
    wandb.config.max_length = max([len(i) for i in data_frame['sequence']])
    data_set = SequenceDataset(data_frame, wandb.config.target, transform=SeqToInt(get_vocab()))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set.scale_targets(scaler)

    # plot_series(data_set.get_targets(), wandb.config.target + ' distribution')

    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=wandb.config.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               drop_last=True)
    return train_loader, scaler


def load_testing_data(collate_fn, scaler):
    file = data_dir + wandb.config.data_set + '_test.tsv'
    data_frame = pd.read_csv(file, sep='\t')[['sequence', wandb.config.target]]
    data_set = SequenceDataset(data_frame, wandb.config.target, transform=SeqToInt(get_vocab()))
    data_set.scale_targets(scaler)

    test_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=wandb.config.batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=True)
    return test_loader


def load_validation_data(collate_fn, scaler):
    file = data_dir + wandb.config.data_set + '_val.tsv'
    data_frame = pd.read_csv(file, sep='\t')[['sequence', wandb.config.target]]

    data_set = SequenceDataset(data_frame, wandb.config.target, transform=SeqToInt(get_vocab()))
    data_set.scale_targets(scaler)

    val_loader = torch.utils.data.DataLoader(dataset=data_set,
                                             batch_size=wandb.config.batch_size,
                                             collate_fn=collate_fn,
                                             shuffle=False,
                                             drop_last=True)
    return val_loader
