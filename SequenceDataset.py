from torch.utils.data import Dataset
from pickle import dump
import wandb


class SequenceDataset(Dataset):
    def __init__(self, data_frame, target_name, transform=None):
        """
            :param data_frame: The pandas dataframe that contains the data
            :param target_name: The name of the target column in the data file.
            :param delimiter (optional): The delimiter separating columns in the data file.
                The delimiter is automatically detected if not provided.
            :param transform (optional): Optional transform to be applied on a sample.
        """

        self.target_name = target_name
        self.sequence_data = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, index):
        seq = self.sequence_data['sequence'][index]
        target = self.sequence_data[self.target_name][index].astype(float)

        if wandb.config.use_charge:
            charge = self.sequence_data['charge'][index]
            sample = {'sequence': seq, 'charge': charge, 'target': target,
                      'length': len(seq)}
        else:
            sample = {'sequence': seq, 'target': target, 'length': len(seq)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_targets(self):
        return self.sequence_data[self.target_name]

    def get_max_length(self):
        return max([len(i) for i in self.sequence_data['sequence']])

    def scale_targets(self, scaler):
        targets = self.sequence_data[self.target_name]
        scaled_targets = scaler.fit_transform(targets.values.reshape(-1, 1))
        # save the scaler so that it can be loaded for evaluation or prediction
        dump(scaler, open('scaler.pkl', 'wb'))
        self.sequence_data[self.target_name] = scaled_targets


'''
class SequenceChargeDataset(Dataset):
    def __init__(self, data_frame, target_name, transform=None):
        """
            :param data_frame: The pandas dataframe that contains the data
            :param target_name: The name of the target column in the data file.
            :param delimiter (optional): The delimiter separating columns in the data file.
                The delimiter is automatically detected if not provided.
            :param transform (optional): Optional transform to be applied on a sample.
        """

        self.target_name = target_name
        self.sequence_data = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, index):
        seq = self.sequence_data['sequence'][index]
        target = self.sequence_data[self.target_name][index].astype(float)
        charge = self.sequence_data['charge'][index]

        sample = {'sequence': seq, 'charge': charge, 'target': target, 'length': len(seq)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_targets(self):
        return self.sequence_data[self.target_name]

    def get_max_length(self):
        return max([len(i) for i in self.sequence_data['sequence']])

    def scale_targets(self, scaler):
        targets = self.sequence_data[self.target_name]
        scaled_targets = scaler.fit_transform(targets.values.reshape(-1, 1))
        # save the scaler so that it can be loaded for evaluation or prediction
        dump(scaler, open('scaler.pkl', 'wb'))
        self.sequence_data[self.target_name] = scaled_targets
'''


class SequencePredDataset(Dataset):
    def __init__(self, data_frame, transform=None):
        """
            :param data_frame: The pandas dataframe that contains the data
            :param delimiter (optional): The delimiter separating columns in the data file.
                The delimiter is automatically detected if not provided.
            :param transform (optional): Optional transform to be applied on a sample.
        """

        self.sequence_data = data_frame
        self.transform = transform

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, index):
        seq = self.sequence_data['sequence'][index]

        if wandb.config.use_charge:
            charge = self.sequence_data['charge'][index]
            sample = {'sequence': seq, 'charge': charge, 'length': len(seq)}
        else:
            sample = {'sequence': seq, 'length': len(seq)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_max_length(self):
        return max([len(i) for i in self.sequence_data['sequence']])
