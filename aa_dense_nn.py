import torch.nn as nn
import torch.nn.functional as F

class AADenseNN(nn.Module):

    def __init__(self):
        super(AADenseNN, self).__init__()
        self.fc1 = nn.Linear(20, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
