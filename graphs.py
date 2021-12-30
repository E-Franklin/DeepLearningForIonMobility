import math
from pathlib import Path
import decimal

import numpy as np
import pandas as pd
from scipy import constants
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import stats
from torch import nn
from sklearn.metrics import mean_squared_error
import wandb
from DataUtils import ccs_to_k0, med_rel_error, delta_t95, delta_tr95, \
    delta_t90_err
from evaluate import run_and_log_stats
from path_config import data_dir

file = 'CNN_419455_pred.tsv'
df = pd.read_csv(f'test_pipeline\\sample_size\\{file}', sep='\t')

# calculate the error % and add as a column
df['error'] = ((df['IM'] - df['Actual']) / df['Actual']) * 100
mre = np.around(med_rel_error(df['Actual'], df['IM']), decimals=1)

num_bin = math.ceil(max(df['error']) - min(df['error']))

fig = px.histogram(df, x="error",
                   barmode='group',
                   labels={
                       'error': 'Peptide CCS prediction Error (%)'
                   },
                   nbins=num_bin,
                   # color of histogram bars
                   color_discrete_sequence=['#006999']
                   )

fig.add_annotation(x=5, y=8000,
                   text=f'Median rel. error (%): {mre}',
                   showarrow=False,
                   yshift=10)

fig.show()
fig.write_image(f'test_pipeline\\sample_size\\CNN_mre_hist.png')
