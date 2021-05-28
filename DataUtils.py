import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_series(data, title=''):
    sns.displot(data, kind='kde')
    plt.subplots_adjust(top=.90)
    plt.title(title, y=1.04)
    plt.savefig('plots/' + title + '.png')
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


def setup_data_loaders(train_dataset, test_dataset, valid_dataset, batch_size, collate_fn):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=None,
                                               batch_sampler=None,
                                               collate_fn=collate_fn,
                                               shuffle=True,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=False,
                                              drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                             batch_size=batch_size,
                                             collate_fn=collate_fn,
                                             shuffle=False,
                                             drop_last=True)

    return train_loader, test_loader, val_loader
