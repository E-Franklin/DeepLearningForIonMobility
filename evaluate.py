import numpy as np
import pandas as pd
import torch
from pathlib import Path

from scipy.stats import stats

import wandb
from DataUtils import delta_t95, delta_tr95, med_rel_error, delta_t90_err
import plotly.express as px
import plotly.graph_objects as go


def evaluate(model, config, data_loader, scaler):
    # store the targets and predictions to be written to a file
    targets_list = []
    preds_list = []

    model.eval()
    with torch.no_grad():
        sum_val_loss_abs = 0
        total = 0
        all_losses = []
        for i, (seqs, charges, targets, lengths) in enumerate(data_loader):
            #seqs = seqs.to(config.device).long()
            targets = targets.view(config.batch_size, 1).to(config.device).float()
            charges = charges.to(config.device)

            # predict outputs
            if config.model_type == 'LSTM':
                outputs = model(seqs, charges, lengths)
            else:
                outputs = model(seqs, charges)

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


def run_and_log_stats(targets_list, preds_list, data_set, plot_eval, model_name, config):
    df = pd.DataFrame({'Actual': targets_list, 'Pred': preds_list},
                      columns=['Actual', 'Pred'])

    delta_95 = delta_t95(df['Actual'], df['Pred'])
    delta_tr_95 = delta_tr95(df['Actual'], df['Pred'])
    med_error = med_rel_error(df['Actual'], df['Pred'])
    delt90_err = delta_t90_err(df['Actual'], df['Pred'])
    pearson_coef, p_value = stats.pearsonr(df['Actual'], df['Pred'])

    wandb.log({f'{data_set}_delta_t95': delta_95, f'{data_set}_delta_tr95': delta_tr_95,
               f'{data_set}_med_rel_error %': med_error,
               f'{data_set}_delta_t90_rel_error %': delt90_err,
               f'{data_set}_pearson_coef': pearson_coef, f'{data_set}_p-value': p_value})

    if plot_eval:

        # create a file to store the targets and predictions
        filename = 'output_data/' + model_name + '_validation_tar_pred.csv'
        exists = Path(filename).exists()
        if exists:
            raise RuntimeError('File ' + filename + ' already exists')

        df.to_csv(filename, index=False)

        residuals = df['Actual'] - df['Pred']
        df['colour'] = ['outlier' if abs(resid) > delta_95 else 'point' for resid in residuals]

        fig = px.scatter(df, x='Actual', y='Pred', color='colour', template='ggplot2',
                         title=f'Actual vs Predicted for model {model_name}',
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

    print(f'{data_set} Evaluation complete. Delta_t95: {delta_95}, Med Rel Err: {med_error}')
    return med_error


def evaluate_model(model, model_name, data_loader, config, data_set, scaler=None, load_model=False, plot_eval=False):
    """
    This evaluate method runs the evaluation loop and saves the targets and prediction as well as plotting the
    Actual vs Predicted values and adding the delta_t95 lines
    """
    print("Begin model eval")

    if load_model:
        model.load_state_dict(torch.load('models/' + model_name + '.pt'))

    targets_list, preds_list = evaluate(model, config, data_loader, scaler)
    med_err = run_and_log_stats(targets_list, preds_list, data_set, plot_eval, model_name, config)

    return med_err
