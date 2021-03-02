import pandas as pd
import numpy as np
from collections import Counter


def encode_seq(seq):
    # define the possible characters in the sequence, - is used for padding
    aas = '-ACDEFGHIKLMNPQRSTVWY'

    # define the mappings for char to int and int to char
    aa_to_int = dict((a, i) for i, a in enumerate(aas))
    int_to_aa = dict((i, a) for i, a in enumerate(aas))

    # use the aa_to_int dict to encode the seq as a list of ints
    seq_encoded = [aa_to_int[char] for char in seq]

    onehot = []
    for i in seq_encoded:
        vec = [1 if val == i else 0 for val in range(len(aas))]
        onehot.append(vec)

    return onehot









def get_counts(s, aacode_list):
    c = Counter(s)
    # subtract makes sure all counters have all keys
    c.subtract(aacode_list)
    return c


def proportion(col):
    return col / max(col)


def add_aa_norm_counts(data):
    # The 2003 paper that used normalized counts, normalized the count of amino acids in a peptide to the highest
    # number of times that amino acid appeared in any peptide in the dataset
    letter_codes = "ACDEFGHIKLMNPQRSTVWY"
    aacode_list = Counter(dict(zip(letter_codes, np.zeros(20))))
    counts = [get_counts(s, aacode_list) for s in data["Sequence"]]
    df = pd.DataFrame(counts, columns=sorted(counts[0].keys()))
    df = df.apply(proportion, axis=0)
    return data.join(df)
