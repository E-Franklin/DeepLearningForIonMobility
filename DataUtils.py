
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from SequenceDataset import *
from SeqToIntTransform import *
from sklearn.preprocessing import MinMaxScaler

from path_config import data_dir


def plot_series(data, title=''):
    sns.displot(data, kind='kde')
    plt.subplots_adjust(top=.90)
    plt.title(title, y=1.04)
    plt.savefig('plots/' + title + '.png')
    #wandb.log({'title': plt})
    plt.show()


def get_stats(data):
    if type(data) is list:
        data = pd.Series(data)
    plot_series(data)
    mean = data.mean()
    stdev = data.std()
    print(f'Mean: {mean:.3f} stdev: {stdev:.3f}')


def plot_losses(losses, title='', params=''):
    g = sns.scatterplot(x=range(len(losses)), y=losses)
    g.figure.subplots_adjust(top=.75)
    g.axes.set_title(title + '\n' + params, y=1.04)
    plt.savefig('plots/' + title + '.png')
    plt.show()


def get_vocab():
    # define the possible characters in the sequence, - is used for padding. a is acetylation and m is methionine oxidation.
    # This will be the same for all datasets as they will be preprocessed. The size of this list affects the encoding.
    aas = '-ACDEFGHIKLMNPQRSTVWYam'

    # define the mappings for char to int and int to char
    vocab = dict((a, i) for i, a in enumerate(aas))
    return vocab


def load_data(data_file):
    data_frame = pd.read_csv(data_dir + data_file, sep='\t')[['sequence', wandb.config.target]]
    wandb.config.max_length = max([len(i) for i in data_frame['sequence']])

    data = SequenceDataset(data_frame, wandb.config.target, transform=SeqToInt(get_vocab()))

    return data


def load_training_data(collate_fn):
    data_set = load_data(wandb.config.data_set + '_train.tsv')

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_set.scale_targets(scaler)

    plot_series(data_set.get_targets(), wandb.config.target + ' distribution')

    train_loader = torch.utils.data.DataLoader(dataset=data_set,
                                               batch_size=wandb.config.batch_size,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               drop_last=True)
    return train_loader, scaler


def load_testing_data(collate_fn, scaler):
    data_set = load_data(wandb.config.data_set + '_test.tsv')
    data_set.scale_targets(scaler)

    test_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=wandb.config.batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=True)
    return test_loader


def load_validation_data(collate_fn, scaler):
    data_set = load_data(wandb.config.data_set + '_val.tsv')
    data_set.scale_targets(scaler)

    val_loader = torch.utils.data.DataLoader(dataset=data_set,
                                             batch_size=wandb.config.batch_size,
                                             collate_fn=collate_fn,
                                             shuffle=False,
                                             drop_last=True)
    return val_loader
