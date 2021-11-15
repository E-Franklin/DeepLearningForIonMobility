from datetime import datetime

import torch

import wandb
from train import train_model

# Check if CUDA is available on the system and use it if so.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the wandb config. The config can be updated before each run
default_config = {
    'device': device,
    # 'embedding_dim': 120,
    'output_size': 1,
    # The value used by Meier et al is 64 and by DeepRT is 16
    'batch_size': 64,
    'num_epochs': 15,
    'learning_rate': 0.001,
    'dropout': 0,
    'data_dir': 'data_sets\\'
}


'''
    # The name of the column containing the values to be predicted.
    'target': 'RT',
    'target_unit': 'min',
    'use_charge': False,
    # 'model_type': The type of model you want to construct. Supports 'LSTM', 'Conv', 'Transformer'
    'model_name': The name that you want the model to have
    # ----- depending on the model type, further parameters need to be set ------
    # LSTM
    'num_lstm_units': 60,
    'num_layers': 2,
    'bidirectional': False,
    'dropout': 0,
    # Conv
    'kernel': 9,
    # pad_by specifies whether to pad all sequences in a data set to a max_length. This is the
    # default when not set. The max_length will be determined and set in the load_data function.
    # To pad each batch to its own max length set pad_by to batch.
    'pad_by': 'batch',
    # this is the file name for the dataset. Suffixes will automatically be added to load the proper data file.
    'data_set':  # 'dia'
                'lab_data'
                # 'deep_learning_ccs'
                # 'deep_learning_ccs_sample'
                # 'deep_learning_ccs_nomod',
'''
'''
# -------- EXP 1a : LSTM RT Prediction on DIA data -----------

model_name = 'LSTM_RT_prediction_deeprt_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config['model_name'] = model_name

config['data_set'] = 'dia'
 
config['target'] = 'RT'
config['target_unit'] = 'min'
config['use_charge'] = False
config['model_type'] = 'LSTM'
config['bidirectional'] = True
config['learning_rate'] = 0.001
config['num_layers'] = 3
config['num_lstm_units'] = 128
config['dropout'] = 0.1
config['pad_by'] = 'batch'

run = wandb.init(project="DeepLearningForIonMobility",
                 name=config['model_name'],
                 config=config,
                 reinit=True)
train_model()

run.finish()
'''

run_configs = []

# -------- EXP 1 : LSTM RT Prediction on Lab data -----------
config_1 = {'data_set': 'lab_data_rt', 'model_type': 'LSTM', 'target': 'RT',
            'target_unit': 'min',
            'use_charge': False, 'bidirectional': True, 'num_layers': 3,
            'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 2 : Convolution RT Prediction on Lab data -----------
config_2 = {'data_set': 'lab_data_rt', 'model_type': 'Conv', 'target': 'RT',
            'target_unit': 'min',
            'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 3 : Transformer RT Prediction on Lab data -----------
config_3 = {'data_set': 'lab_data_rt', 'model_type': 'Transformer',
            'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2,
            'dim_feed_fwd': 200, 'pad_by': 'batch'}

# -------- EXP 4 : LSTM RT Prediction on Deep Learning CCS Data -----------
config_4 = {'data_set': 'deep_learning_ccs_rt', 'model_type': 'LSTM',
            'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'bidirectional': True, 'num_layers': 3,
            'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 5 : Convolution RT Prediction on Deep Learning CCS Data -----------
config_5 = {'data_set': 'deep_learning_ccs_rt', 'model_type': 'Conv',
            'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 6 : Transformer RT Prediction on Deep Learning CCS Data -----------
config_6 = {'data_set': 'deep_learning_ccs_rt', 'model_type': 'Transformer',
            'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2,
            'dim_feed_fwd': 200, 'pad_by': 'batch'}

# -------- EXP 7 : LSTM IM Prediction on Lab Data Without Charge -----------
config_7 = {'data_set': 'lab_data_im_nc', 'model_type': 'LSTM', 'target': 'IM',
            'target_unit': '1/K0',
            'use_charge': False, 'bidirectional': True, 'num_layers': 3,
            'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 8 : Convolution IM Prediction on Lab Data Without Charge -----------
config_8 = {'data_set': 'lab_data_im_nc', 'model_type': 'Conv', 'target': 'IM',
            'target_unit': '1/K0',
            'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 9 : Transformer IM Prediction on Lab Data Without Charge -----------
config_9 = {'data_set': 'lab_data_im_nc', 'model_type': 'Transformer',
            'target': 'IM', 'target_unit': '1/K0',
            'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2,
            'dim_feed_fwd': 200, 'pad_by': 'batch'}

# -------- EXP 10 : LSTM IM Prediction on Lab Data With Charge -----------
config_10 = {'data_set': 'lab_data_im_wc', 'model_type': 'LSTM', 'target': 'IM',
             'target_unit': '1/K0',
             'use_charge': True, 'bidirectional': True, 'num_layers': 3,
             'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 11 : Convolution IM Prediction on Lab Data With Charge -----------
config_11 = {'data_set': 'lab_data_im_wc', 'model_type': 'Conv', 'target': 'IM',
             'target_unit': '1/K0',
             'use_charge': True, 'kernel': 10, 'pad_by': ''}

# -------- EXP 12 : Transformer IM Prediction on Lab Data With Charge -----------
config_12 = {'data_set': 'lab_data_im_wc', 'model_type': 'Transformer',
             'target': 'IM', 'target_unit': '1/K0',
             'use_charge': True, 'num_attn_heads': 2, 'num_layers': 2,
             'dim_feed_fwd': 200, 'embedding_dim': 512, 'pad_by': 'none'}

# -------- EXP 13 : LSTM IM Prediction on Deep Learning CCS Data Without Charge -----------
config_13 = {'data_set': 'deep_learning_ccs_im_nc', 'model_type': 'LSTM',
             'target': 'IM', 'target_unit': 'A',
             'use_charge': False, 'bidirectional': True, 'num_layers': 3,
             'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 14 : Convolution IM Prediction on Deep Learning CCS Data Without Charge -----------
config_14 = {'data_set': 'deep_learning_ccs_im_nc', 'model_type': 'Conv',
             'target': 'IM', 'target_unit': 'A',
             'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 15 : Transformer IM Prediction on Deep Learning CCS Data Without Charge -----------
config_15 = {'data_set': 'deep_learning_ccs_im_nc', 'model_type': 'Transformer',
             'target': 'IM', 'target_unit': 'A',
             'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2,
             'dim_feed_fwd': 200, 'pad_by': 'batch'}

# -------- EXP 16 : LSTM IM Prediction on Deep Learning CCS Data With Charge -----------
config_16 = {'data_set': 'deep_learning_ccs_im_wc', 'model_type': 'LSTM',
             'target': 'IM', 'target_unit': 'A',
             'use_charge': True, 'bidirectional': True, 'num_layers': 3,
             'num_lstm_units': 128, 'embedding_dim': 512, 'pad_by': 'none'}

# -------- EXP 17 : Convolution IM Prediction on Deep Learning CCS Data With Charge -----------
config_17 = {'data_set': 'deep_learning_ccs_im_wc', 'model_type': 'Conv',
             'target': 'IM', 'target_unit': 'A',
             'use_charge': True, 'kernel': 10, 'embedding_dim': 512, 'pad_by': ''}

# -------- EXP 18 : Transformer IM Prediction on Deep Learning CCS Data With Charge -----------
config_18 = {'data_set': 'deep_learning_ccs_im_wc', 'model_type': 'Transformer',
             'target': 'IM', 'target_unit': 'A',
             'use_charge': True, 'num_attn_heads': 2, 'num_layers': 2,
             'dim_feed_fwd': 200, 'embedding_dim': 512, 'pad_by': 'none'}

# TODO: Make runs with deepRT data
'''
# --------- EXP 19: LSTM DeepRT Lab data ------------
config_1 = {'data_set': 'lab_data_rt', 'model_type': 'LSTM', 'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'bidirectional': True, 'num_layers': 3, 'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 2 : Convolution RT Prediction on Lab data -----------
config_2 = {'data_set': 'lab_data_rt', 'model_type': 'Conv', 'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 3 : Transformer RT Prediction on Lab data -----------
config_3 = {'data_set': 'lab_data_rt', 'model_type': 'Transformer', 'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2, 'dim_feed_fwd': 200, 'pad_by': 'batch'}

# -------- EXP 4 : LSTM RT Prediction on Deep Learning CCS Data -----------
config_4 = {'data_set': 'deep_learning_ccs_rt', 'model_type': 'LSTM', 'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'bidirectional': True, 'num_layers': 3, 'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 5 : Convolution RT Prediction on Deep Learning CCS Data -----------
config_5 = {'data_set': 'deep_learning_ccs_rt', 'model_type': 'Conv', 'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 6 : Transformer RT Prediction on Deep Learning CCS Data -----------
config_6 = {'data_set': 'deep_learning_ccs_rt', 'model_type': 'Transformer', 'target': 'RT', 'target_unit': 'min',
            'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2, 'dim_feed_fwd': 200, 'pad_by': 'batch'}
'''

# -------- EXP 7 : LSTM IM Prediction on Lab Data Without Charge -----------
config_19 = {'data_set': 'lab_data_im_nc_ccs', 'model_type': 'LSTM', 'target': 'CCS',
             'target_unit': 'A',
             'use_charge': False, 'bidirectional': True, 'num_layers': 3,
             'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 8 : Convolution IM Prediction on Lab Data Without Charge -----------
config_20 = {'data_set': 'lab_data_im_nc_ccs', 'model_type': 'Conv', 'target': 'CCS',
             'target_unit': 'A',
             'use_charge': False, 'kernel': 10, 'pad_by': ''}

# -------- EXP 9 : Transformer IM Prediction on Lab Data Without Charge -----------
config_21 = {'data_set': 'lab_data_im_nc_ccs', 'model_type': 'Transformer',
             'target': 'CCS', 'target_unit': 'A',
             'use_charge': False, 'num_attn_heads': 2, 'num_layers': 2,
             'dim_feed_fwd': 200, 'pad_by': 'batch'}

# -------- EXP 10 : LSTM IM Prediction on Lab Data With Charge -----------
config_22 = {'data_set': 'lab_data_im_wc_ccs', 'model_type': 'LSTM', 'target': 'CCS',
             'target_unit': 'A',
             'use_charge': True, 'bidirectional': True, 'num_layers': 3,
             'num_lstm_units': 128, 'pad_by': 'batch'}

# -------- EXP 11 : Convolution IM Prediction on Lab Data With Charge -----------
config_23 = {'data_set': 'lab_data_im_wc_ccs', 'model_type': 'Conv', 'target': 'CCS',
             'target_unit': 'A',
             'use_charge': True, 'kernel': 10, 'pad_by': ''}

# -------- EXP 12 : Transformer IM Prediction on Lab Data With Charge -----------
config_24 = {'data_set': 'lab_data_im_wc_ccs', 'model_type': 'Transformer',
             'target': 'CCS', 'target_unit': 'A',
             'use_charge': True, 'num_attn_heads': 2, 'num_layers': 2,
             'dim_feed_fwd': 200, 'pad_by': 'batch'}

# config_1, config_2, config_3, config_4, config_5, config_6, config_7, config_8, config_9, config_10, config_11, config_12,
#                              config_13, config_14, config_15, config_16, config_17, config_18
run_configs = run_configs + [config_16]  # config_17, config_18 config_19, config_20, config_21, config_22, config_23, config_24

replicate = 1
for conf in run_configs:
    for i in range(replicate):
        model_name = f"{conf['model_type']}_{conf['target']}_{conf['data_set']}_" \
                     f"{datetime.now().strftime('%Y%m%d-%H%M')}_rep{i}"
        conf['model_name'] = model_name

        config = default_config
        config.update(conf)

        run = wandb.init(project="DeepLearningForIonMobility",
                         name=model_name,
                         config=config,
                         dir=output_dir,
                         reinit=True)
        train_model()

        run.finish()

exit()
