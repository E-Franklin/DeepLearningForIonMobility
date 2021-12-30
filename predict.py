from pickle import load

import torch

import wandb
from Collate import pred_collate
from DataUtils import load_pred_data
from train import build_model


def predict_values(data_file, model_path, scaler_path,
                   output_dir, transform=None):
    """
    :param data_file: File containing the data to be used for prediction.
    Must contain a "squence" column and optionally "charge".
    :param model_path: The path to the model file
    :param scaler_path: The path to the scaler pkl file
    :param output_dir: Directory where ouput will be written
    :param transform: the function to use to transform the outputs (ex.
    ccs_to_k0)
    :return: None
    """

    # create a data loader for the data file
    data_frame, data_loader = load_pred_data(pred_collate, data_file)

    # build the model
    model = build_model(wandb.config)

    # load the model weights
    model.load_state_dict(torch.load(model_path))
    if scaler_path:
        # load the scaler
        scaler = load(open('scaler.pkl', 'rb'))

    # predict the ouput values
    preds_list = []

    # set the model to eval mode
    model.eval()
    with torch.no_grad():
        for i, (seqs, charges, lengths) in enumerate(data_loader):
            seqs = seqs.to(wandb.config.device).long()
            charges = charges.to(wandb.config.device)

            # predict outputs
            if wandb.config.model_type == 'LSTM':
                outputs = model(seqs, charges, lengths)
            else:
                outputs = model(seqs, charges)

            outputs = outputs.data.cpu().numpy()
            if scaler_path:
                outputs = scaler.inverse_transform(outputs)

            preds_list.append(outputs.squeeze())
            #preds_list.extend(outputs.squeeze().tolist())


    # write the original data out to a file with a column for the new values
    if transform:
        data_frame[wandb.config.target] = transform(data_frame['PrecursorMz'],
                                                    data_frame[
                                                        'charge'],
                                                    preds_list)
    else:
        data_frame[wandb.config.target] = preds_list

    data_frame.to_csv(f'{output_dir}/'
                      f'{wandb.config.model_type}_prediction_output'
                      f'.tsv',
                      sep='\t',
                      index=False)
