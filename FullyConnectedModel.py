import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    def __init__(self, input_size, batch_size, device, output_size=1):
        super(FCModel, self).__init__()

        self.batch_size = batch_size
        self.device = device

        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, output_size)

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)

        return out3
