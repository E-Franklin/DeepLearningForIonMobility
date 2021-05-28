import math

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import DataUtils
from Collate import pad_sort_collate
from RTLSTM import RTLSTM
from RTLSTMOnehot import *
from SeqToIntTransform import *
from SequenceDataset import *
from datetime import datetime

from evaluate import evaluate_model
from pathlib import Path
import wandb


wandb.init(project="LSTM Training")
# will train on GPU CUDA cores if they are available in the system
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
wandb.config.input_size = 20
num_lstm_units = 60
num_layers = 2
output_size = 1
batch_size = 4
num_epochs = 4
learning_rate = 0.01

do_train = True
evaluate = True

use_onehot = True
scaled = False
use_min = False

model_name = 'LSTM_onehot_' + datetime.now().strftime("%Y%m%d-%H%M%S")

# model_name = 'Onehot_no_scaling'
# model_name = 'LSTM_Onehot_scaling'
# model_name = 'LSTM_Embed_no_scaling'
# model_name = 'LSTM_Embed_scaling'


def get_param_string():
    rt_unit = 'sec'
    if use_min:
        rt_unit = 'min'

    pstring = 'Onehot: ' + str(use_onehot) + \
              ' input_size: ' + str(wandb.config.input_size) + \
              ', num_lstm_units: ' + str(num_lstm_units) + \
              ', num_layers: ' + str(num_layers) + ',\n' + \
              'output_size: ' + str(output_size) + \
              ', batch_size: ' + str(batch_size) + \
              ', num_epochs: ' + str(num_epochs) + \
              ', learning_rate: ' + str(learning_rate) + ',\n' + \
              'scaled: ' + str(scaled) + \
              ', rt_unit: ' + rt_unit
    return pstring


# define the possible characters in the sequence, - is used for padding
aas = '-ACDEFGHIKLMNPQRSTVWY'

# define the mappings for char to int and int to char
vocab = dict((a, i) for i, a in enumerate(aas))
to_int = SeqToInt(vocab)

target_name = 'RT'
data_frame = pd.read_csv('data\\2021-03-12-easypqp-frac-lib-openswath_processed.tsv', sep='\t')[['sequence', target_name]]
if use_min:
    data_frame[target_name] = data_frame[target_name]/60

rt_data = SequenceDataset(data_frame, target_name, transform=to_int)

scaler = None
if scaled:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rt_data.scale_targets(scaler)

DataUtils.plot_series(rt_data.get_targets(), 'RT distribution')

num_data_points = len(rt_data)

train_split = math.floor(0.8 * num_data_points)
test_split = math.floor(0.1 * num_data_points)
val_split = num_data_points - train_split - test_split

train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(rt_data,
                                                                           [train_split, test_split, val_split],
                                                                           generator=torch.Generator().manual_seed(42))

if use_onehot:
    model = RTLSTMOnehot(wandb.config.input_size, num_lstm_units, num_layers, batch_size, vocab, device).to(device)
else:
    model = RTLSTM(wandb.config.input_size, num_lstm_units, num_layers, batch_size, vocab, device).to(device)

train_loader, test_loader, val_loader = DataUtils.setup_data_loaders(train_dataset, test_dataset, valid_dataset,
                                                                     batch_size, pad_sort_collate)

if do_train:

    def mse(pred, tar):
        return np.mean((tar - pred) ** 2)


    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # file to store the targets and predictions for further analysis
    # The file name uses the model name. Raises an exception and does not train if the file already exists.
    filename = 'data/' + model_name + '_training_tar_pred.csv'
    exists = Path(filename).exists()
    if exists:
        raise RuntimeError('File ' + filename + ' already exists')

    # store the targets and predictions to be written to a file
    targets_list = []
    preds_list = []

    # store the losses and the unscaled losses (if the data has been scaled, so that scaled and unscaled losses will be
    # comparable). The losses are summed so that they can be averaged over 100 batches to plot the loss.
    losses = []
    losses_unscaled = []
    sum_loss = 0
    unscaled_sum_loss = 0
    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (seqs, targets, lengths) in enumerate(train_loader):
            seqs = seqs.to(device).long()
            targets = targets.view(batch_size, 1).to(device).float()

            # Forward pass
            outputs = model(seqs, lengths)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()

            numpy_targets = targets.data.cpu().numpy()
            numpy_outputs = outputs.data.cpu().numpy()

            if scaled:
                unscaled_out = scaler.inverse_transform(numpy_outputs)
                unscaled_tar = scaler.inverse_transform(numpy_targets)
                unscaled_sum_loss += mse(unscaled_out, unscaled_tar)

            targets_list.extend(numpy_targets.squeeze().tolist())
            preds_list.extend(numpy_outputs.squeeze().tolist())

            if (i + 1) % 100 == 0:
                losses.append(sum_loss / 100)
                wandb.log({'loss': sum_loss / 100})
                losses_unscaled.append(unscaled_sum_loss / 100)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, sum_loss / 100))
                sum_loss = 0
                unscaled_sum_loss = 0

    # write the targets and predicted values to a file for later analysis.
    df = pd.DataFrame({'Actual': targets_list, 'Pred': preds_list},
                      columns=['Actual', 'Pred'])
    df.to_csv(filename, index=False)

    DataUtils.plot_losses(losses_unscaled, model_name + ' Training Loss Unscaled', get_param_string())

    DataUtils.plot_losses(losses, model_name + ' Training Loss', get_param_string())
    for epoch in range(num_epochs):
        start = epoch * math.floor(total_step / 100)
        end = (epoch + 1) * math.floor((total_step / 100))
        DataUtils.plot_losses(losses[start:end],
                              model_name + 'Epoch ' + str(epoch + 1) + ' loss', get_param_string())

    torch.save(model.state_dict(), 'models/' + model_name + '.pt')

if evaluate:
    evaluate_model(model, model_name, scaled, scaler, device, batch_size, val_loader)
