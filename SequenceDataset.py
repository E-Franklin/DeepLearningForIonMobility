from torch.utils.data import Dataset


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

        sample = {'sequence': seq, 'target': target, 'length': len(seq)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_targets(self):
        return self.sequence_data[self.target_name]

    def scale_targets(self, scaler):
        targets = self.sequence_data[self.target_name]
        scaled_targets = scaler.fit_transform(targets.values.reshape(-1, 1))
        self.sequence_data[self.target_name] = scaled_targets
