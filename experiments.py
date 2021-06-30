from datetime import datetime

import torch

import wandb
from train import train

# Check if CUDA is available on the system and use it if so.
from train_transformer import train_transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the wandb config. The config can be updated before each run
config = {
    'device': device,
    'embedding_dim': 22,
    'output_size': 1,
    'batch_size': 4,
    'num_epochs': 4,
    'learning_rate': 0.01,
    # The name of the column containing the values to be predicted.
    'target': 'RT',
    'target_unit': 'min',
    'use_charge': False,
    # The type of model you want to construct. Supports 'FC', 'LSTM', 'Conv'
    # 'model_type': 'LSTM',
    'model_name': '',
    # depending on the model type, further parameters need to be set
    # LSTM
    'num_lstm_units': 60,
    'num_layers': 2,
    'bidirectional': False,
    # Conv
    'kernel': 9,
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

    'plot_eval': True
}

'''
# -------- EXP 1 : LSTM RT Prediction on Lab data -----------

model_name = 'LSTM_RT_prediction_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.005
config['num_layers'] = 3
config['num_lstm_units'] = 100
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 2 : LSTM RT Prediction on Deep Learning CCS Data -----------

model_name = 'LSTM_RT_prediction_dl_ccs_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.005
config['num_layers'] = 3
config['num_lstm_units'] = 100
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()


# -------- EXP 3 : LSTM IM Prediction on Lab Data Without Charge -----------

model_name = 'LSTM_IM_prediction_no_charge_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'IM'
config['target_unit'] = '1/K0'
config['use_charge'] = False
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.005
config['num_layers'] = 3
config['num_lstm_units'] = 100
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 4 : LSTM IM Prediction on Lab Data With Charge -----------

model_name = 'LSTM_IM_prediction_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'IM'
config['target_unit'] = '1/K0'
config['use_charge'] = True
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.005
config['num_layers'] = 3
config['num_lstm_units'] = 100
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 5 : LSTM IM Prediction on Deep Learning CCS Data Without Charge -----------

model_name = 'LSTM_IM_prediction_dl_ccs_data_no_charge_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'IM'
config['target_unit'] = 'A'
config['use_charge'] = False
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.005
config['num_layers'] = 3
config['num_lstm_units'] = 100
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 6 : LSTM IM Prediction on Deep Learning CCS Data With Charge -----------

model_name = 'LSTM_IM_prediction_dl_ccs_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'IM'
config['target_unit'] = 'A'
config['use_charge'] = True
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.005
config['num_layers'] = 3
config['num_lstm_units'] = 100
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()
'''

'''
# -------- EXP 7 : Convolution RT Prediction on Lab data -----------

model_name = 'Conv_RT_prediction_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Conv'
config['kernel'] = 10
config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = ''

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 8 : Convolution RT Prediction on Deep Learning CCS Data -----------

model_name = 'Conv_RT_prediction_dl_ccs_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Conv'
config['kernel'] = 10
config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = ''

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()


# -------- EXP 9 : Convolution IM Prediction on Lab Data Without Charge -----------

model_name = 'Conv_IM_prediction_no_charge_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Conv'
config['kernel'] = 10
config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'IM'
config['target_unit'] = '1/K0'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = ''

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 10 : Convolution IM Prediction on Lab Data With Charge -----------

model_name = 'Conv_IM_prediction_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Conv'
config['kernel'] = 10
config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'IM'
config['target_unit'] = '1/K0'
config['use_charge'] = True
config['learning_rate'] = 0.001
config['pad_by'] = ''

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 11 : Convolution IM Prediction on Deep Learning CCS Data Without Charge -----------

model_name = 'Conv_IM_prediction_dl_ccs_data_no_charge_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Conv'
config['kernel'] = 10
config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'IM'
config['target_unit'] = 'A'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = ''

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()

# -------- EXP 12 : Convolution IM Prediction on Deep Learning CCS Data With Charge -----------

model_name = 'Conv_IM_prediction_dl_ccs_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Conv'
config['kernel'] = 10
config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'IM'
config['target_unit'] = 'A'
config['use_charge'] = True
config['learning_rate'] = 0.001
config['pad_by'] = ''

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train()

run.finish()
'''


# -------- EXP 13 : Transformer RT Prediction on Lab data -----------
model_name = 'Transformer_RT_pred_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Transformer'
config['num_attn_heads'] = 2
config['num_layers'] = 2
config['dim_feed_fwd'] = 200
config['dropout'] = 0
config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_transformer()

run.finish()

# -------- EXP 14 : Transformer RT Prediction on Deep Learning CCS Data -----------

model_name = 'Transformer_RT_pred_dl_ccs_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Transformer'
config['num_attn_heads'] = 2
config['num_layers'] = 2
config['dim_feed_fwd'] = 200
config['dropout'] = 0
config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_transformer()

run.finish()


# -------- EXP 15 : Transformer IM Prediction on Lab Data Without Charge -----------

model_name = 'Transformer_IM_pred_no_charge_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Transformer'
config['num_attn_heads'] = 2
config['num_layers'] = 2
config['dim_feed_fwd'] = 200
config['dropout'] = 0
config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'IM'
config['target_unit'] = '1/K0'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_transformer()

run.finish()

# -------- EXP 16 : Transformer IM Prediction on Lab Data With Charge -----------

model_name = 'Transformer_IM_pred_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Transformer'
config['num_attn_heads'] = 2
config['num_layers'] = 2
config['dim_feed_fwd'] = 200
config['dropout'] = 0
config['data_set'] = 'lab_data'
config['batch_size'] = 4
config['target'] = 'IM'
config['target_unit'] = '1/K0'
config['use_charge'] = True
config['learning_rate'] = 0.001
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_transformer()

run.finish()

# -------- EXP 17 : Transformer IM Prediction on Deep Learning CCS Data Without Charge -----------

model_name = 'Transformer_IM_pred_dl_ccs_data_no_charge_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Transformer'
config['num_attn_heads'] = 2
config['num_layers'] = 2
config['dim_feed_fwd'] = 200
config['dropout'] = 0
config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'IM'
config['target_unit'] = 'A'
config['use_charge'] = False
config['learning_rate'] = 0.001
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_transformer()

run.finish()

# -------- EXP 18 : Transformer IM Prediction on Deep Learning CCS Data With Charge -----------

model_name = 'Transformer_IM_pred_dl_ccs_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['model_type'] = 'Transformer'
config['num_attn_heads'] = 2
config['num_layers'] = 2
config['dim_feed_fwd'] = 200
config['dropout'] = 0
config['data_set'] = 'deep_learning_ccs'
config['batch_size'] = 10
config['target'] = 'IM'
config['target_unit'] = 'A'
config['use_charge'] = True
config['learning_rate'] = 0.001
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_transformer()

run.finish()