import pandas as pd
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import MinMaxScaler

class SequenceDataset(Dataset):
    def __init__(self, data_file, target_name, delimiter=None, transform=None):
        """
            :param data_file: The path to the file containing the sequence and target data.
            :param target_name: The name of the target column in the data file.
            :param delimiter (optional): The delimiter separating columns in the data file.
                The delimiter is automatically detected if not provided.
            :param transform (optional): Optional transform to be applied on a sample.
        """

        self.target_name = target_name
        # TODO pass in engine='python' in delim is None to avoid warning
        self.sequence_data = pd.read_csv(data_file, sep=delimiter)[['sequence', target_name]]
        self.transform = transform

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, index):

        seq = self.sequence_data['sequence'][index]
        target = self.sequence_data[self.target_name][index].astype(float)

        sample = {'sequence': seq, 'target': target, 'length': len(seq)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_targets(self):
        return self.sequence_data[self.target_name]

    def scale_targets(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        targets = self.sequence_data[self.target_name]
        scaled_targets = scaler.fit_transform(targets.values.reshape(-1, 1))
        self.sequence_data[self.target_name] = scaled_targets
