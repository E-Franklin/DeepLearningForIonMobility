import torch
import torch.nn as nn
import SequenceDataset as dl
import encoding as enc
import aa_dense_nn as ann
import torch.optim as optim
import math

loader = dl.DataLoader('data\\')
evidence_data = loader.load_file('annotated1K0_evidence.csv')

# get the rows for peptides that do not have modifications
unmodified_peptides = evidence_data[evidence_data.Modifications == 'Unmodified']

rt_data = unmodified_peptides[
            ['Sequence', 'Experiment', 'Retention time', 'Charge', 'PEP', 'IonMobilityIndexK0']
            ]

rt_data_sorted = rt_data.sort_values(by='Experiment', axis=0)
rt_data_sorted.isna().sum()
rt_data_sorted = rt_data_sorted.dropna().reset_index(drop=True)

# sort by the posterior error probability and then drop duplicate rows
# keep the ones with the lowest error (since sort is ascending we keep the first one)
print(len(rt_data_sorted[rt_data_sorted.Experiment == 1]))
rt_data_exp1 = rt_data_sorted[rt_data_sorted.Experiment == 1]\
                .sort_values('PEP')\
                .drop_duplicates(subset=['Sequence', 'Charge'], keep='first')\
                .reset_index(drop=True)
print(len(rt_data_exp1))
rt_data_exp1 = enc.add_aa_norm_counts(rt_data_exp1)

input_columns = list('ACDEFGHIKLMNPQRSTVWY')
inputs = rt_data_exp1[input_columns].values
inputs = torch.tensor(inputs).float()
targets = torch.tensor(rt_data_exp1['Retention time'].values).float()

rt_dataset = torch.utils.data.TensorDataset(inputs, targets)
num_items = len(rt_data_exp1)
train_size = math.floor(num_items*0.6)
validate_size = math.floor(num_items*0.2)
test_size = num_items - train_size - validate_size

train, validate, test = torch.utils.data.random_split(
                            rt_dataset,
                            [train_size,
                                validate_size,
                                test_size])
BATCH_SIZE = 4
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

net = ann.AADenseNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

EPOCHS = 50
for e in range(EPOCHS):
    cumulative_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        targets = (targets.view(-1, 1))

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        if i % 1000 == 999:
            print(f'{e+1}, {i+1}, loss: {cumulative_loss/1000:.3f}')
            cumulative_loss = 0.0

print('Training complete!')
