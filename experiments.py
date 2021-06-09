from datetime import datetime

import torch

import wandb

from train import train

# Check if CUDA is available on the system and use it if so.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the wandb config. The config can be updated before each run
config = {
    'device': device,
    'embedding_dim': 22,
    'num_lstm_units': 60,
    'num_layers': 2,
    'output_size': 1,
    'batch_size': 4,
    'num_epochs': 4,
    'learning_rate': 0.01,
    # The name of the column containing the values to be predicted.
    'target': 'RT',
    # The type of model you want to construct. Supports 'FC', 'LSTM', 'Convolution'
    'model_type': 'LSTM',
    'model_name': '',
    # pad_by specifies whether to pad all sequences in a data set to a max_length. This is the
    # default when not set. The max_length will be determined and set in the load_data function.
    # To pad each batch to its own max length set pad_by to batch.
    'pad_by': 'batch',
    'data_dir': 'data_sets\\',
    # this is the file name for the dataset. Suffixes will automatically be added to load the proper data file.
    'data_set':  # 'dia'
                'lab_data'
                # 'deep_learning_ccs'
                # 'deep_learning_ccs_nomod'
}

# Experiment example
model_name = 'LSTM_RT_prediction_lab_data_test1_' + datetime.now().strftime("%Y%m%d-%H%M%S")
# initialize wandb run
wandb.init(project="DeepLearningForIonMobility",
           name=config['model_name'],
           config=config)
train()

'''
if use_onehot:
    pass
else:
    model = RTLSTM(wandb.config.embedding_dim, wandb.config.num_lstm_units, wandb.config.num_layers, vocab).to(device)
'''
