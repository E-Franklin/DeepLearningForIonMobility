"""
Split the original data file into the train, test, and validate sets and save them to separate files.
This will allow the data sets to be loaded independently and used for multiple models.
"""
import sys

import numpy as np
import pandas as pd
import re


def split_data(path, fracs=None, subsample=False, sample_size=None, use_base=True):
    data_frame = pd.read_csv(path, sep='\t')
    # shuffle the input
    data_frame = data_frame.sample(frac=1, random_state=1)
    if subsample:
        data_frame = data_frame.sample(n=sample_size, random_state=1)

    # set the fractions for the train, test, validate split
    if fracs is None:
        fracs = [0.7, 0.15, 0.15]

    fractions = np.array(fracs)

    if use_base:
        # take the base sequence
        base = data_frame['base_sequence']
        print(len(base))
        # filter to only the unique base sequences
        base = base.drop_duplicates()
        print(len(base))
        # split the sequences into train, test, and validate sets
        train_a, test_a, val_a = np.array_split(base, (fractions[:-1].cumsum() * len(base)).astype(int))
        # join the base sequences back to the original data
        train = data_frame.merge(train_a, how='right', on='base_sequence')
        test = data_frame.merge(test_a, how='right', on='base_sequence')
        val = data_frame.merge(val_a, how='right', on='base_sequence')
    else:
        # split into 3 parts
        train, test, val = np.array_split(data_frame, (fractions[:-1].cumsum() * len(data_frame)).astype(int))

    print(f'train: {len(train)}, test:{len(test)}, val:{len(val)}')

    # remove the file extension from the input data path
    output = re.sub(r'\..*', '', path)

    # write out the tree data set parts to separate files
    train.to_csv(output + '_train.tsv', sep='\t', index=False)
    test.to_csv(output + '_test.tsv', sep='\t', index=False)
    val.to_csv(output + '_val.tsv', sep='\t', index=False)


# TODO: additional error checking
# TODO: use flags
# allow data file to be specified from command line
if len(sys.argv) == 2:
    data_path = sys.argv[1]
    split_data(data_path)
elif len(sys.argv) == 4:
    data_path = sys.argv[1]
    sub_sample = sys.argv[2]
    n = int(sys.argv[3])
    split_data(data_path, sub_sample, n)
else:
    print('Usage: split_data.py <data_path> <sub_sample:boolean=False> <sample_size:int=None>')
