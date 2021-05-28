import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import data_dir
import numpy as np

val_data = pd.read_csv(data_dir + 'LSTM_Onehot_No_scaling_20210426-204204_validation_tar_pred.csv')

val_data['resid'] = val_data['Actual'] - val_data['Pred']

quants = val_data.quantile(q=[0.025, 0.975], axis=0)
min_interval = quants['resid'][0.025]
max_interval = quants['resid'][0.975]
val_data['colours'] = np.where(val_data['resid'] >= min_interval or val_data['resid'] <= max_interval, 1, 0)

graph = sns.scatterplot(x='Actual', y='Pred', hue='colours', ci=None, data=val_data)
plt.title('Observed Vs Predicted RT', fontsize=25)
plt.xlabel('Observed RT (s)', fontsize=20)
plt.ylabel('Predicted RT (s)', fontsize=20)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], color='red', transform=ax.transAxes)
plt.grid(True)
plt.show()
