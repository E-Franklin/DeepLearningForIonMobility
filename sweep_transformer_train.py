from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch import nn

import wandb
from Collate import pad_sort_collate
from DataUtils import load_training_data, get_vocab
from DataUtils import load_validation_data, delta_t95
from transformer import SequenceTransformer

# Check if CUDA is available on the system and use it if so.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the wandb config. These are the defaults for the sweep. Some will be overwritten by the sweep.
config_default = {
    'device': device,
    'embedding_dim': 200,
    'output_size': 1,
    'batch_size': 4,
    'num_epochs': 4,
    'learning_rate': 0.001,
    # The name of the column containing the values to be predicted.
    'target': 'RT',
    'target_unit': 'min',
    'use_charge': False,
    # The type of model you want to construct. Supports 'FC', 'LSTM', 'Conv'
    'model_type': 'Transformer',
    'model_name': '',
    # depending on the model type, further parameters need to be set
    # Transformer
    'num_attn_heads': 2,
    'num_layers': 2,
    'dim_feed_fwd': 200,
    'dropout': 0,
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

# initialize wandb run
model_name = 'Transformer_RT_prediction_lab_data_' + datetime.now().strftime("%Y%m%d-%H%M%S")
config_default['model_name'] = model_name
wandb.init(config=config_default)


def evaluate_transformer(model, config, collate_fn, scaler):
    # store the targets and predictions to be written to a file
    targets_list = []
    preds_list = []

    val_loader = load_validation_data(collate_fn, scaler)

    model.eval()
    with torch.no_grad():
        sum_val_loss_abs = 0
        total = 0
        all_losses = []
        for i, (seqs, targets, lengths) in enumerate(val_loader):
            seqs = seqs.to(config.device).long()
            targets = targets.view(config.batch_size, 1).to(config.device).float()

            src_mask = model.generate_square_subsequent_mask(len(seqs[0])).to(device)

            # Forward pass
            outputs = model(seqs, src_mask)

            targets = targets.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()

            if scaler is not None:
                targets = scaler.inverse_transform(targets)
                outputs = scaler.inverse_transform(outputs)

            sum_val_loss_abs += np.sum(abs(outputs - targets))
            all_losses = np.append(all_losses, abs(outputs - targets))
            total += len(outputs)

            targets_list.extend(targets.squeeze().tolist())
            preds_list.extend(outputs.squeeze().tolist())

    return targets_list, preds_list


def evaluate_transformer_model(model, tmodel_name, collate_fn, config, scaler=None, load_model=False):
    """
    This evaluate method runs the evaluation loop and saves the targets and prediction as well as plotting the
    Actual vs Predicted values and adding the delta_t95 lines
    """
    print("Begin model eval")
    # if load_model is true then load the model parameters from a saved model,
    # otherwise the model passed is a model that has already been trained
    if load_model:
        model.load_state_dict(torch.load('models/' + tmodel_name + '.pt'))

    # create a file to store the targets and predictions for further analysis
    # The file name uses the model name. Raises an exception and does not run if the file already exists.
    filename = 'output_data/' + tmodel_name + '_validation_tar_pred.csv'
    exists = Path(filename).exists()
    if exists:
        raise RuntimeError('File ' + filename + ' already exists')

    targets_list, preds_list = evaluate_transformer(model, config, collate_fn, scaler)

    # write the targets and predicted values to a file for later analysis.
    df = pd.DataFrame({'Actual': targets_list, 'Pred': preds_list},
                      columns=['Actual', 'Pred'])
    df.to_csv(filename, index=False)

    # Get the delta_t95 and build the plot
    delta_95 = delta_t95(df['Actual'], df['Pred'])
    residuals = df['Actual'] - df['Pred']
    df['colour'] = ['outlier' if abs(resid) > delta_95 else 'point' for resid in residuals]

    fig = px.scatter(df, x='Actual', y='Pred', color='colour', template='ggplot2',
                     title=f'Actual vs Predicted values from the validation set for model {tmodel_name}',
                     labels={
                         'Actual': f'Actual {config["target"]} ({config["target_unit"]})',
                         'Pred': f'Predicted {config["target"]} ({config["target_unit"]})'
                     })
    fig.add_trace(
        go.Scatter(
            x=df['Actual'],
            y=df['Actual'] - delta_95,
            mode='lines',
            line=go.scatter.Line(color='gray'),
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['Actual'],
            y=df['Actual'] + delta_95,
            mode='lines',
            line=go.scatter.Line(color='gray'),
            showlegend=False
        )
    )

    wandb.log({'Act vs Pred plot': fig})
    wandb.log({'delta_t95': delta_95})
    print(f'Evaluation complete. Delta_t95: {delta_95}')


def train_transformer():
    config = wandb.config

    # load the training data
    collate_fn = pad_sort_collate
    train_loader, scaler = load_training_data(collate_fn)

    # Define the model
    print('Building Transformer model')
    model = SequenceTransformer(get_vocab(), config.embedding_dim, config.num_attn_heads, config.dim_feed_fwd,
                                config.num_layers, config.dropout).to(config.device)

    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    sum_loss = 0

    for epoch in range(config.num_epochs):
        for i, (seqs, targets, lengths) in enumerate(train_loader):
            seqs = seqs.to(config.device).long()
            targets = targets.view(config.batch_size, 1).to(config.device).float()

            src_mask = model.generate_square_subsequent_mask(len(seqs[0])).to(device)

            # Forward pass
            outputs = model(seqs, src_mask)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            if (i + 1) % 100 == 0:
                wandb.log({'epoch': epoch, 'loss': sum_loss / 100})
                sum_loss = 0
    print('Training complete. Saving model ' + config.model_name)
    torch.save(model.state_dict(), 'models/' + config.model_name + '.pt')
    print('Model saved.')

    evaluate_transformer_model(model, config.model_name, collate_fn, config, scaler)


train_transformer()
