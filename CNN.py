import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding import onehot_encoding
import wandb


# CNN
class CNN(nn.Module):
    def __init__(self, conv1_kernel, use_charge, vocab, embedding_dim=20):
        super(CNN, self).__init__()

        self.use_charge = use_charge
        self.embedding_dim = embedding_dim

        self.encoder = nn.Embedding(num_embeddings=(len(vocab)),
                                    embedding_dim=embedding_dim,
                                    padding_idx=vocab['-'])

        # following the structure of DeepRT but without the capsules since that is a custom layer and I would like to
        # compare to a vanilla CNN. DeepRT:
        # https://github.com/horsepurve/DeepRTplus/blob/master/capsule_network_emb.py#L243
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(embedding_dim, conv1_kernel), stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, conv1_kernel), stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.out_size = (wandb.config.max_length - 2 * conv1_kernel) + 2

        self.fc = nn.Linear(in_features=256 * 1 * self.out_size, out_features=60)
        self.fc_charge = nn.Linear(in_features=256 * 1 * self.out_size + 1, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=20)
        self.out = nn.Linear(in_features=20, out_features=1)

    def forward(self, x, charges):
        x = self.encoder(x)

        x = x.transpose(dim0=1, dim1=2)  # -> [batch, dict, len]
        x = x[:, None, :, :]  # -> [batch, 1, dict, len]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        # flatten the output from the CNN
        x = x.view(-1, 256 * 1 * self.out_size)

        #if self.use_charge:
        #    x = torch.cat([x, charges], dim=1)
        #    x = self.fc_charge(x)
        #else:
        x = self.fc(x)

        x = F.relu(self.fc2(x))
        out = self.out(x)

        return out
