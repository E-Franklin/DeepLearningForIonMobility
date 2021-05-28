import pandas as pd

data = pd.read_csv('data\\2021-03-12-easypqp-frac-lib-openswath_new.tsv', sep='\t')

# filter out the decoys
data = data[data['Decoy'] == 0]

# select relevant columns and drop duplicate rows
data = data[['PrecursorCharge', 'NormalizedRetentionTime', 'PeptideSequence', 'PrecursorIonMobility']]

data.rename(columns={'NormalizedRetentionTime': 'RT', 'PeptideSequence': 'sequence', 'PrecursorIonMobility': 'IM'}, inplace=True)
data.drop_duplicates(keep='first', inplace=True)
data.reset_index(inplace=True, drop=True)

data.to_csv('data\\2021-03-12-easypqp-frac-lib-openswath_processed.tsv', sep='\t', index=False)
