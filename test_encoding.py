import encoding
import numpy as np

# TODO: fix this test. The types being checked are not correct as we no longer use numpy
def test_onehot_encoding():
    seq = 'ACDEFGHIKLMNPQRSTVWY-'
    expected_output = np.identity(21)
    actual_output = encoding.onehot_encoding(seq)

    assert len(expected_output) == len(actual_output)
    assert np.array_equal(expected_output, actual_output)
