import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_series(data):
    sns.displot(data, kind='kde', rug=True)
    plt.show()


def get_stats(data):
    if type(data) is list:
        data = pd.Series(data)
    plot_series(data)
    mean = data.mean()
    stdev = data.std()
    print(f'Mean: {mean:.3f} stdev: {stdev:.3f}')
