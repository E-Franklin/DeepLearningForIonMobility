

class SeqToInt:
    """
    Convert a sequence of tokens to integers

    Args:
        vocab: a dictionary containing the mapping of tokens to ints
    """
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, sample):
        seq, target, length = sample['sequence'], sample['target'], sample['length']

        seq = [self.vocab[char] for char in seq]

        return {'sequence': seq, 'target': target, 'length': length}
