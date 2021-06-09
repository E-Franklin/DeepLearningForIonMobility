import numpy as np
import pandas as pd
import torch
from pathlib import Path
import wandb
from DataUtils import load_validation_data


def evaluate_model(model, model_name, collate_fn, scaler=None):
    print("Begin model eval")
    model.load_state_dict(torch.load('models/' + model_name + '.pt'))

    # create a file to store the targets and predictions for further analysis
    # The file name uses the model name. Raises an exception and does not train if the file already exists.
    filename = 'output_data/' + model_name + '_validation_tar_pred.csv'
    exists = Path(filename).exists()
    if exists:
        raise RuntimeError('File ' + filename + ' already exists')

    # store the targets and predictions to be written to a file
    targets_list = []
    preds_list = []

    val_loader = load_validation_data(collate_fn)

    model.eval()
    with torch.no_grad():
        sum_val_loss_abs = 0
        total = 0
        all_losses = []
        for i, (seqs, targets, lengths) in enumerate(val_loader):
            seqs = seqs.to(wandb.config.device).long()
            targets = targets.view(wandb.config.batch_size, 1).to(wandb.config.device).float()

            # predict outputs
            outputs = model(seqs, lengths)

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

    # write the targets and predicted values to a file for later analysis.
    df = pd.DataFrame({'Actual': targets_list, 'Pred': preds_list},
                      columns=['Actual', 'Pred'])
    df.to_csv(filename, index=False)

    table = wandb.Table(dataframe=df)

    # TODO: Will need to generate the entire figure and then log it to wandb since you cannot plot multiple types of data in one plot
    wandb.log({'actual_vs_pred': wandb.plot.scatter(table, x='Actual', y='Pred'), 'line_plot': wandb.plot.line(table, x='Actual', y='Actual', stroke='Pred')})
    wandb.log({'Average Loss MAE': sum_val_loss_abs / total, 'Med Abs Error': np.median(all_losses)})
