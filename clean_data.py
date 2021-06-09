import pandas as pd
import re

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

data = pd.read_csv('data_sets\\SourceData_Figure_1.csv', sep=',')
data.rename(columns={'Modified sequence': 'sequence', 'Sequence': 'base sequence', 'Retention time': 'RT', 'CCS': 'IM'}, inplace=True)

# count the number of peptides
print(len(data))

# create a no mods dataset from the CCS prediction one.
# remove the _ from the start and end of each modified sequence
data['sequence'] = data['sequence'].str.replace('_', '')


# Take the rows where the modified and unmodified sequences match
data_no_mods = data[data['sequence'] == data['base sequence']]

data_no_mods.to_csv('data_sets\\CCS_prediction_SourceData_Figure1_no_modified_peptides.tsv', sep='\t', index=False)
'''
# Replace the modifications with new letters for amino acids.
mod_patterns = []
for i in data['Modified sequence']:
    mod_patterns.extend(re.findall(r'.\(ox\)', i))


unique_mods = set(mod_patterns)

print(unique_mods)
'''

# Replace the markers for acetylation and methionine oxidation with the single character markers that will be used in encoding
data['sequence'] = data['sequence'].str.replace(r'\(ac\)', 'a', regex=True)
data['sequence'] = data['sequence'].str.replace(r'M\(ox\)', 'm', regex=True)

data.to_csv('data_sets\\DeepLearningCCS_SourceDataFig1_processed.tsv', sep='\t', index=False)
print("Finished data processing")
