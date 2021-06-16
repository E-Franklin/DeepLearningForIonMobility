import numpy as np
import pandas as pd
import torch
from pathlib import Path
import wandb
from DataUtils import load_validation_data, delta_t95, delta_tr95
import plotly.express as px
import plotly.graph_objects as go


def evaluate(model, config, collate_fn, scaler):
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

    return targets_list, preds_list


def evaluate_model(model, model_name, collate_fn, config, scaler=None, load_model=False):
    """
    This evaluate method runs the evaluation loop and saves the targets and prediction as well as plotting the
    Actual vs Predicted values and adding the delta_t95 lines
    """
    print("Begin model eval")
    # if load_model is true then load the model parameters from a saved model,
    # otherwise the model passed is a model that has already been trained
    if load_model:
        model.load_state_dict(torch.load('models/' + model_name + '.pt'))

    # create a file to store the targets and predictions for further analysis
    # The file name uses the model name. Raises an exception and does not run if the file already exists.
    filename = 'output_data/' + model_name + '_validation_tar_pred.csv'
    exists = Path(filename).exists()
    if exists:
        raise RuntimeError('File ' + filename + ' already exists')

    targets_list, preds_list = evaluate(model, config, collate_fn, scaler)

    # write the targets and predicted values to a file for later analysis.
    df = pd.DataFrame({'Actual': targets_list, 'Pred': preds_list},
                      columns=['Actual', 'Pred'])
    df.to_csv(filename, index=False)

    # Get the delta_t95 and build the plot
    delta_95 = delta_t95(df['Actual'], df['Pred'])
    residuals = df['Actual'] - df['Pred']
    df['colour'] = ['outlier' if abs(resid) > delta_95 else 'point' for resid in residuals]

    fig = px.scatter(df, x='Actual', y='Pred', color='colour', template='ggplot2',
                     title=f'Actual vs Predicted values from the validation set for model {model_name}',
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


def evaluate_model_quick(model, collate_fn, config, scaler=None):
    """
    This is a shorter form of the evaluate_model method that runs the evaluation loop
    but does not write targets and predicted values to a file or plot the values.
    :param model:
    :param collate_fn:
    :param config:
    :param scaler:
    :return:
    """
    print("Begin model eval")

    # run the eval loop
    targets_list, preds_list = evaluate(model, config, collate_fn, scaler)
    df = pd.DataFrame({'Actual': targets_list, 'Pred': preds_list},
                      columns=['Actual', 'Pred'])

    # Get the delta_t95 and build the plot
    delta_95 = delta_t95(df['Actual'], df['Pred'])
    delta_tr_95 = delta_tr95(df['Actual'], df['Pred'])

    wandb.log({'delta_t95': delta_95, 'delta_tr95': delta_tr_95})
    print(f'Evaluation complete. Delta_t95: {delta_95}')
