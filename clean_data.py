import pandas as pd
import re
import math
from pandas import DataFrame
from scipy import constants


def k0_to_ccs(df: DataFrame):
    # constants
    n0 = constants.value(
        u'Loschmidt constant (273.15 K, 101.325 kPa)')  # Loschmidt's number
    kB = constants.value(u'Boltzmann constant')  # JK-1
    uamu = constants.value(u'unified atomic mass unit')  # 1/12 12C in kg
    e = constants.value(
        u'elementary charge')  # elementary charge C = A*s (Ampere*second)
    t = 305  # K
    # t0 = 273.15  # K
    # p = 2.7  # mbar, 0.002664692820133235 atm
    # p0 = 1013.25  # mbar, 1 atm
    mg = 28  # mass of N2 in Da

    ccs_values = []

    for j in range(len(df)):
        # calculate the CCS from 1/k0
        mi = df.loc[j, 'Mass']
        mu_kg = ((mi * mg) / (mi + mg)) * uamu  # convert Da to kg
        term2 = math.sqrt((2 * constants.pi) / (mu_kg * kB * t))

        q = df.loc[j, 'PrecursorCharge']
        term3 = (q * e) / (
                (1 / df.loc[j, 'PrecursorIonMobility'] * 10 ** -4) * n0)

        ccs = (3 / 16) * term2 * term3 * 10 ** 20  # 10**20 converts m**2 to Angstrom
        print(ccs)
        # print(data.loc[i, 'CCS'])

        ccs_values.append(ccs)

    return ccs_values


data_set_lengths = {}

# --------------------------------------------------------------------------------------------------------------
# Lab Data

lab_data = pd.read_csv(
    'data_sets\\2021-03-12-easypqp-frac-lib-openswath_new.tsv', sep='\t')

# filter out the decoys
lab_data = lab_data[lab_data['Decoy'] == 0]
lab_data = lab_data.reset_index(drop=True)

# calculate the mass and add it as a column
lab_data['Mass'] = lab_data.loc[:, 'PrecursorMz'] * lab_data.loc[:, 'PrecursorCharge']

# calculate the ccs values and add them to the data frame
ccs_val = k0_to_ccs(lab_data)
lab_data['CCS'] = ccs_val

# select relevant columns and rename them
lab_data = lab_data.loc[:,
                        ['PrecursorCharge', 'NormalizedRetentionTime', 'PeptideSequence',
                         'ModifiedPeptideSequence', 'PrecursorIonMobility', 'CCS']]

lab_data.rename(
    columns={'PrecursorCharge': 'charge', 'NormalizedRetentionTime': 'RT',
             'PeptideSequence': 'base_sequence',
             'ModifiedPeptideSequence': 'sequence',
             'PrecursorIonMobility': 'IM'}, inplace=True)
# find all of the modifications and print so that we can see if there are mods that are not captured by the replace
mod_patterns = []
for i in lab_data['sequence']:
    mod_patterns.extend(re.findall(r'.\(.[^\)]*\)', i))

unique_mods = set(mod_patterns)
print(unique_mods)

# replace the modifications with the single letters that are used to represent them
lab_data['sequence'] = lab_data['sequence'].str.replace(r'\.\(UniMod:1\)', 'a',
                                                        regex=True)
lab_data['sequence'] = lab_data['sequence'].str.replace(r'C\(UniMod:4\)', 'c',
                                                        regex=True)
lab_data['sequence'] = lab_data['sequence'].str.replace(r'M\(UniMod:35\)', 'm',
                                                        regex=True)
'''
# ---------------------------------------
# Prepare the data set for RT prediction
# ---------------------------------------

rt_pred_lab_set = lab_data.loc[:, ['base_sequence', 'sequence', 'RT']]
# make unique on modified sequence
rt_pred_lab_set.drop_duplicates(subset='sequence', keep='first', inplace=True)
rt_pred_lab_set.reset_index(inplace=True, drop=True)

# write to a file to be split later
rt_pred_lab_set.to_csv('data_sets\\lab_data_rt.tsv', sep='\t', index=False)
data_set_lengths['lab_data_rt'] = len(rt_pred_lab_set)

# prepare the data set for DeepRT
deep_rt_lab_set = lab_data.loc[:, ['base_sequence', 'RT']]
deep_rt_lab_set.rename(columns={'base_sequence': 'sequence'}, inplace=True)

# make unique on sequence
deep_rt_lab_set.drop_duplicates(subset='sequence', keep='first', inplace=True)
deep_rt_lab_set.reset_index(inplace=True, drop=True)

# write to a file to be split later
deep_rt_lab_set.to_csv('data_sets\\lab_data_deeprt.tsv', sep='\t', index=False)
data_set_lengths['lab_data_deeprt'] = len(deep_rt_lab_set)
'''
# ------------------------------------
# Prepare the data for IM prediction
# ------------------------------------

im_pred_set_wc = lab_data.loc[:, ['base_sequence', 'sequence', 'charge', 'IM', 'CCS']]
# make unique on modified sequence
im_pred_set_wc.drop_duplicates(subset=['sequence', 'charge'], keep='first',
                               inplace=True)
im_pred_set_wc.reset_index(inplace=True, drop=True)

# write to a file to be split later
im_pred_set_wc.to_csv('data_sets\\lab_data_im_wc_ccs.tsv', sep='\t',
                      index=False)
data_set_lengths['lab_data_im_wc'] = len(im_pred_set_wc)

# data set for learning im without charge
im_pred_set_nc = lab_data.loc[:, ['base_sequence', 'sequence', 'charge', 'IM', 'CCS']]
im_pred_set_nc = im_pred_set_nc.sort_values('charge')
# make unique on modified sequence
im_pred_set_nc.drop_duplicates(subset=['sequence'], keep='first', inplace=True)
im_pred_set_nc.reset_index(inplace=True, drop=True)

# write to a file to be split later
im_pred_set_nc.to_csv('data_sets\\lab_data_im_nc_ccs.tsv', sep='\t',
                      index=False)
data_set_lengths['lab_data_im_nc'] = len(im_pred_set_nc)
'''
# ----------------------------------------------------------------------------
# Deep Learning CCS data

data_deep_ccs = pd.read_csv('data_sets\\SourceData_Figure_1.csv', sep=',')

data_deep_ccs.rename(columns={'Sequence': 'base_sequence', 'Modified sequence': 'sequence',
                     'Charge': 'charge', 'Retention time': 'RT', 'CCS': 'IM'}, inplace=True)
data_deep_ccs = data_deep_ccs[['base_sequence', 'sequence', 'charge', 'RT', 'IM', 'Experiment', 'Score']]

# count the number of peptides
print(len(data_deep_ccs))


# remove the _ from the start and end of each modified sequence

# Take the rows where the modified and unmodified sequences match
data_no_mods = data[data['sequence'] == data['base sequence']]
data_no_mods.to_csv('data_sets\\CCS_prediction_SourceData_Figure1_no_modified_peptides.tsv', sep='\t', index=False)


# Replace the modifications with new letters for amino acids.
mod_patterns = []
for i in data_deep_ccs['sequence']:
    mod_patterns.extend(re.findall(r'.\(.[^\)]*\)', i))

unique_mods = set(mod_patterns)
print(unique_mods)

# Replace the markers for acetylation and methionine oxidation with the single character markers that will be used in encoding
data_deep_ccs['sequence'] = data_deep_ccs['sequence'].str.replace(r'\(ac\)', 'a', regex=True)
data_deep_ccs['sequence'] = data_deep_ccs['sequence'].str.replace(r'M\(ox\)', 'm', regex=True)
data_deep_ccs['sequence'] = data_deep_ccs['sequence'].str.replace('_', '')

data_deep_ccs = data_deep_ccs.sort_values('Score', ascending=False)

# ---------------------------------------
# Prepare the data set for RT prediction
# ---------------------------------------

# Filter out all rows belonging to experiment HeLa_Trp_2 which has 120119 peptides
data_deep_ccs_rt_pred = data_deep_ccs[data_deep_ccs['Experiment'] == 'HeLa_Trp_2']

# ########### Prep DeepRT data set
data_deep_ccs_deeprt = data_deep_ccs_rt_pred.drop_duplicates(subset='base_sequence', keep='first')
data_deep_ccs_deeprt = data_deep_ccs_deeprt[['base_sequence', 'RT']]
data_deep_ccs_deeprt.rename(columns={'base_sequence': 'sequence'}, inplace=True)
data_deep_ccs_deeprt.to_csv('data_sets\\deep_learning_ccs_deeprt.tsv', sep='\t', index=False)
data_set_lengths['deep_learning_ccs_deeprt'] = len(data_deep_ccs_deeprt)

# ########### Prep RT prediction
data_deep_ccs_rt_pred = data_deep_ccs_rt_pred.drop_duplicates(subset='sequence', keep='first')
data_deep_ccs_rt_pred = data_deep_ccs_rt_pred[['base_sequence', 'sequence', 'RT']]
data_deep_ccs_rt_pred.to_csv('data_sets\\deep_learning_ccs_rt.tsv', sep='\t', index=False)
data_set_lengths['deep_learning_ccs_rt'] = len(data_deep_ccs_rt_pred)

# ---------------------------------------
# Prepare the data set for IM prediction
# ---------------------------------------
data_deep_ccs_im_wc = data_deep_ccs.drop_duplicates(subset=['sequence', 'charge'], keep='first')
data_deep_ccs_im_wc = data_deep_ccs_im_wc[['base_sequence', 'sequence', 'charge', 'IM']]
data_deep_ccs_im_wc.to_csv('data_sets\\deep_learning_ccs_im_wc.tsv', sep='\t', index=False)
data_set_lengths['deep_learning_ccs_im_wc'] = len(data_deep_ccs_im_wc)

data_deep_ccs_im_nc = data_deep_ccs.drop_duplicates(subset=['sequence'], keep='first')
data_deep_ccs_im_nc = data_deep_ccs_im_nc[['base_sequence', 'sequence', 'charge', 'IM']]
data_deep_ccs_im_nc.to_csv('data_sets\\deep_learning_ccs_im_nc.tsv', sep='\t', index=False)
data_set_lengths['deep_learning_ccs_im_nc'] = len(data_deep_ccs_im_nc)

with open('data_sets\\data_set_lengths.txt', 'w') as f:
    print(data_set_lengths, file=f)
'''
print("Done processing data")
