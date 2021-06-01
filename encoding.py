
import numpy as np
import torch
import wandb


def onehot_encoding(x, embed_dim):
    encoding = []
    # ident will be a matrix 21x20 with a diagonal of ones starting in the second row
    # padding index assumed to be zero
    ident = np.eye(embed_dim + 1, embed_dim, k=-1, dtype=np.float32)
    for seq in x:
        encoding.append(ident[seq.cpu()])
    return torch.tensor(encoding).to(wandb.config.device)
