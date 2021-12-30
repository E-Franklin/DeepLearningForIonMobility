

class SeqToInt:
    """
    Convert a sequence of tokens to integers

    Args:
        vocab: a dictionary containing the mapping of tokens to ints
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, sample):

        seq = [self.vocab[char] for char in sample['sequence']]
        sample['sequence'] = seq

        return sample


class ChargeSeqToInt:
    """
    Convert a sequence of tokens to integers

    Args:
        vocab: a dictionary containing the mapping of tokens to ints
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, sample):

        seq = [self.vocab[char + str(sample['charge'])] for char in sample['sequence']]
        sample['sequence'] = seq

        return sample
