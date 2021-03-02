import encoding
import numpy as np


def test_onehot_encoding():
    seq = 'ACDEFGHIKLMNPQRSTVWY-'
    expected_output = np.identity(21)
    actual_output = encoding.encode_seq(seq)

    assert len(expected_output) == len(actual_output)
    assert np.array_equal(expected_output, actual_output)
