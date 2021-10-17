import torch
import numpy as np
import wandb
from CPCProt.tokenizer import Tokenizer


def pad_sort_collate(batch):
    """
    :param batch: A batch of samples of the form seq:target:length
    :return: the padded batch and lengths of the sequences
    """
    # get the sequence and targets as lists
    if wandb.config.use_charge:
        batch_seqs, batch_charges, batch_targets, lens = [s['sequence'] for s in
                                                          batch], \
                                                         [s['charge'] for s in
                                                          batch], \
                                                         [s['target'] for s in
                                                          batch], \
                                                         [s['length'] for s in
                                                          batch]
    else:
        batch_seqs, batch_targets, lens = [s['sequence'] for s in batch], \
                                          [s['target'] for s in batch], \
                                          [s['length'] for s in batch]
        batch_charges = [-1 for _ in range(len(batch))]

    # pad the sequences
    if wandb.config.pad_by == 'batch':
        max_length = max(lens)
    elif wandb.config.pad_by == 'max':
        # use the max_length for the data set rather than the batch
        max_length = wandb.config.max_length

    padded_seqs = np.zeros((len(batch), max_length), dtype=int)

    for i, seq_length in enumerate(lens):
        padded_seqs[i, 0:seq_length] = batch_seqs[i][0:seq_length]

    # convert to tensors
    padded_seqs = torch.tensor(padded_seqs)
    batch_charges = torch.tensor(batch_charges)
    batch_charges = batch_charges[:, None]
    batch_targets = torch.tensor(batch_targets)
    batch_lengths = torch.tensor(lens)

    # sort the sequences in the batch from longest to shortest
    batch_lengths, perm_idx = batch_lengths.sort(0, descending=True)
    padded_seqs = padded_seqs[perm_idx]
    batch_targets = batch_targets[perm_idx]

    return padded_seqs, batch_charges, batch_targets, batch_lengths


def collate(batch):
    """
    :param batch: A batch of samples of the form seq:target:length
    :return: the padded batch and lengths of the sequences
    """
    # get the sequence and targets as lists
    if wandb.config.use_charge:
        batch_seqs, batch_charges, batch_targets, lens = [s['sequence'] for s in
                                                          batch], \
                                                         [s['charge'] for s in
                                                          batch], \
                                                         [s['target'] for s in
                                                          batch], \
                                                         [s['length'] for s in
                                                          batch]
    else:
        batch_seqs, batch_targets, lens = [s['sequence'] for s in batch], \
                                          [s['target'] for s in batch], \
                                          [s['length'] for s in batch]
        batch_charges = [-1 for _ in range(len(batch))]

    tokenizer = Tokenizer()
    # convert to tensors
    # Tokenize and convert to torch tensor
    final_seqs = []
    max_len = max(lens)
    for seq in batch_seqs:
        seq_length = len(seq)
        seq = seq.upper()  # this will treat the modified amino acids as unmodified
        encode = tokenizer.encode(seq)
        padded_seq = np.zeros((max_len,), dtype=int)

        padded_seq[0:seq_length] = encode[0:seq_length]

        final_seqs.append(torch.tensor([padded_seq]))

    batch_charges = torch.tensor(batch_charges)
    batch_charges = batch_charges[:, None]
    batch_targets = torch.tensor(batch_targets)
    batch_lengths = torch.tensor(lens)

    return final_seqs, batch_charges, batch_targets, batch_lengths
