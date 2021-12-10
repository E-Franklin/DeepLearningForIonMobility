from datetime import datetime

import torch

import wandb
from train import train_model

# Check if CUDA is available on the system and use it if so.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the wandb config. These are the defaults for the sweep. Some will be overwritten by the sweep.
default_config = {
    'device': device,
    # 'embedding_dim': 120,
    'output_size': 1,
    # The value used by Meier et al is 64 and by DeepRT is 16
    'batch_size': 16,
    'num_epochs': 20,
    'learning_rate': 0.001,
    'dropout': 0,
    'data_dir': 'data_sets\\',
    'data_set': 'lab_data_rt',
    # 'data_set': 'deep_learning_ccs_rt'
    'model_type': 'LSTM',
    'target': 'RT',
    'target_unit': 'min',
    'use_charge': False,
    'pad_by': 'batch'
}

# initialize wandb run
model_name = f"LSTM_{default_config['target']}_{default_config['data_set']}_" \
                     f"{datetime.now().strftime('%Y%m%d-%H%M')}"
default_config['model_name'] = model_name
wandb.init(config=default_config)
train_model()
