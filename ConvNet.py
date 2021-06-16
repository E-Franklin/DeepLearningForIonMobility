import torch.nn as nn
import torch.nn.functional as F
from encoding import onehot_encoding
import wandb


# CNN
class ConvNet(nn.Module):
    def __init__(self, conv1_kernel, embedding_dim=20):
        super(ConvNet, self).__init__()

        self.embedding_dim = embedding_dim

        # following the structure of DeepRT but without the capsules since that is a custom layer and I would like to
        # compare to a vanilla conv net. DeepRT: https://github.com/horsepurve/DeepRTplus/blob/master/capsule_network_emb.py#L243
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(embedding_dim, conv1_kernel), stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, conv1_kernel), stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.out_size = (wandb.config.max_length - 2*conv1_kernel) + 2
        self.fc = nn.Linear(in_features=256*1*self.out_size, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=20)
        self.out = nn.Linear(in_features=20, out_features=1)

    # lengths is taken in here because it comes from the batch collate function but it isn't used.
    # May write a new collate for convnet that doesn't return lengths and always pads to max dataset length
    def forward(self, x, x_lengths):
        # x will have padded sequences
        x = onehot_encoding(x, self.embedding_dim)
        x = x.transpose(dim0=1, dim1=2)  # -> [batch, dict, len]
        x = x[:, None, :, :]  # -> [batch, 1, dict, len]

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        # flatten the output from the CNN
        x = x.view(-1, 256*1*self.out_size)

        x = self.fc(x)
        x = F.relu(self.fc2(x))
        out = self.out(x)

        return out
