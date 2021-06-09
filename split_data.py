"""
Split the original data file into the train, test, and validate sets and save them to separate files.
This will allow the data sets to be loaded independently and used for multiple models.
"""
import sys

import numpy as np
import pandas as pd
import re


def split_data(path):
    data_frame = pd.read_csv(path, sep='\t')

    # set the fractions for the train, test, validate split
    fractions = np.array([0.7, 0.2, 0.1])
    # shuffle the input
    data_frame = data_frame.sample(frac=1)
    # split into 3 parts
    train, test, val = np.array_split(data_frame, (fractions[:-1].cumsum() * len(data_frame)).astype(int))

    # remove the file extension from the input data path
    output = re.sub(r'\..*', '', path)

    # write out the tree data set parts to separate files
    train.to_csv(output + '_train.tsv', sep='\t', index=False)
    test.to_csv(output + '_test.tsv', sep='\t', index=False)
    val.to_csv(output + '_val.tsv', sep='\t', index=False)


# allow data file to be specified from command line
data_path = sys.argv[1]
split_data(data_path)
