import torch
import torch.nn as nn

# will train on GPU CUDA cores if they are available in the system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


