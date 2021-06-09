
import numpy as np
import torch
import wandb


def onehot_encoding(x, embedding_dim):
    encoding = []
    # ident will be a matrix 23x22 with a diagonal of ones starting in the second row
    # padding index assumed to be zero
    ident = np.eye(embedding_dim + 1, embedding_dim, k=-1, dtype=np.float32)
    for seq in x:
        encoding.append(ident[seq.cpu()])
    return torch.tensor(encoding).to(wandb.config.device)
