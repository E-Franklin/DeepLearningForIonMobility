import pandas as pd

'''
data = pd.read_csv('data\\2021-03-12-easypqp-frac-lib-openswath_new.tsv', sep='\t')

# filter out the decoys
data = data[data['Decoy'] == 0]

# select relevant columns and drop duplicate rows
data = data[['PrecursorCharge', 'NormalizedRetentionTime', 'PeptideSequence', 'PrecursorIonMobility']]

data.rename(columns={'NormalizedRetentionTime': 'RT', 'PeptideSequence': 'sequence', 'PrecursorIonMobility': 'IM'}, inplace=True)
data.drop_duplicates(keep='first', inplace=True)
data.reset_index(inplace=True, drop=True)

data.to_csv('data\\2021-03-12-easypqp-frac-lib-openswath_processed.tsv', sep='\t', index=False)
'''

data = pd.read_csv('data\\SourceData_Figure_1.csv', sep=',')

# count the number of modified peptides
print(len(data))
print(len(data[data['Modified sequence'] != data['Sequence']]))

# create a no mods dataset from the CCS prediction one.
# remove the _ from the start and end of each modified sequence
data['Modified sequence'] = [i[1:-1] for i in data['Modified sequence']]

# Take the rows where the modified and unmodified sequences match
data = data[data['Modified sequence'] == data['Sequence']]
