from datetime import datetime

import torch

import wandb
from train import train

# Check if CUDA is available on the system and use it if so.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the wandb config. These are the defaults for the sweep. Some will be overwritten by the sweep.
config_default = {
    'device': device,
    'embedding_dim': 22,
    'output_size': 1,
    'batch_size': 4,
    'num_epochs': 4,
    'learning_rate': 0.01,
    # The name of the column containing the values to be predicted.
    'target': 'RT',
    'target_unit': 'min',
    # The type of model you want to construct. Supports 'FC', 'LSTM', 'Conv'
    'model_type': 'LSTM',
    'model_name': '',
    # depending on the model type, further parameters need to be set
    # LSTM
    'num_lstm_units': 60,
    'num_layers': 2,
    'bidirectional': True,
    # pad_by specifies whether to pad all sequences in a data set to a max_length. This is the
    # default when not set. The max_length will be determined and set in the load_data function.
    # To pad each batch to its own max length set pad_by to batch.
    'pad_by': 'batch',
    'data_dir': 'data_sets\\',
    # this is the file name for the dataset. Suffixes will automatically be added to load the proper data file.
    'data_set':  # 'dia',
                'lab_data',
                # 'deep_learning_ccs',
                # 'deep_learning_ccs_nomod',
    'plot_eval': False
}

# initialize wandb run
model_name = 'LSTM_RT_prediction_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config_default['model_name'] = model_name
wandb.init(config=config_default)
train()
