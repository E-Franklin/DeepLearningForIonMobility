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
    device = torch.device('cpu')

    # load the yaml file and get a python dictionary
    #with open(config_file, 'r') as file:
    #    config = yaml.safe_load(file)

    # build the model
    model = build_model(wandb.config)

    # load the model weights
    model.load_state_dict(torch.load(model_path))
    if scaler_path:
        # load the scaler
        scaler = load(open('scaler.pkl', 'rb'))

    # set the model to eval mode
    model.eval()

    # create a data loader for the data file
    data_frame, data_loader = load_pred_data(pred_collate, data_file)

    # predict the ouput values
    preds_list = []

    with torch.no_grad():
        for i, (seqs, charges, lengths) in enumerate(data_loader):
            seqs = seqs.to(device).long()
            charges = charges.to(wandb.config.device)

            # predict outputs
            if wandb.config.model_type == 'LSTM':
                outputs = model(seqs, charges, lengths)
            else:
                outputs = model(seqs, charges)

            outputs = outputs.data.cpu().numpy()
            if scaler_path:
                outputs = scaler.inverse_transform(outputs)

            preds_list.extend(outputs.squeeze().tolist())


    # write the original data out to a file with a column for the new values
    if transform:
        data_frame[wandb.config.target] = transform(data_frame['PrecursorMz'],
                                                    data_frame[
                                                        'charge'],
                                                    preds_list)
    else:
        data_frame[wandb.config.target] = preds_list

    data_frame.to_csv(f'{output_dir}/prediction_ouput.csv', index=False)
