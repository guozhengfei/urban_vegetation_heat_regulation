import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

CETs = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_extreme_v1.csv')

climate = pd.read_csv(current_dir + '/1_Input/urban_cores_newtowns/urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

merged_df = pd.merge(CETs, climate, on='ID')

# tropical
merged_df[merged_df['koppen'] == 1]['ts_tree'].mean()
merged_df[merged_df['koppen'] == 1]['ts_grass'].mean()
merged_df[merged_df['koppen'] == 1]['ts_cropland'].mean()

# arid
merged_df[merged_df['koppen']==2]['ts_tree'].mean()
merged_df[merged_df['koppen']==2]['ts_grass'].mean()
merged_df[merged_df['koppen']==2]['ts_cropland'].mean()

# temporal
merged_df[merged_df['koppen']==3]['ts_tree'].mean()
merged_df[merged_df['koppen']==3]['ts_grass'].mean()
merged_df[merged_df['koppen']==3]['ts_cropland'].mean()

# boreal
merged_df[merged_df['koppen']==4]['ts_tree'].mean()
merged_df[merged_df['koppen']==4]['ts_grass'].mean()
merged_df[merged_df['koppen']==4]['ts_cropland'].mean()

from brokenaxes import brokenaxes

fig, ax = plt.subplots(1,figsize=(9,4))
bax = brokenaxes(ylims=((0, 0.5), (3.5, 8.5)), hspace=0.05)
x = np.array([0.9, 1, 1.1,1.2])
bax.bar(x, merged_df[merged_df['koppen']==1].iloc[:,[7,2,4,6]].mean().values, yerr=merged_df[merged_df['koppen']==1].iloc[:,[7,2,4,6]].std()*0.25, width=0.1,color=['lightgray','#7fc97f','#beaed4','#fdc086']) # tropical
bax.bar(x+0.5, merged_df[merged_df['koppen']==3].iloc[:,[7,2,4,6]].mean(), yerr=merged_df[merged_df['koppen']==3].iloc[:,[7,2,4,6]].std()*0.25, width=0.1,color=['lightgray','#7fc97f','#beaed4','#fdc086']) # temporal
bax.bar(x+1.0, merged_df[merged_df['koppen']==4].iloc[:,[7,2,4,6]].mean(), yerr=merged_df[merged_df['koppen']==4].iloc[:,[7,2,4,6]].std()*0.25, width=0.1,color=['lightgray','#7fc97f','#beaed4','#fdc086']) # boreal
bax.bar(x+1.5, merged_df[merged_df['koppen']==2].iloc[:,[7,2,4,6]].mean(), yerr=merged_df[merged_df['koppen']==2].iloc[:,[7,2,4,6]].std()*0.25, width=0.1,color=['lightgray','#7fc97f','#beaed4','#fdc086']) # arid
bax.bar(x+2.0, merged_df.iloc[:,[7,2,4,6]].mean(), yerr=merged_df.iloc[:,[7,2,4,6]].std()*0.25, width=0.1,color=['lightgray','#7fc97f','#beaed4','#fdc086']) # global

# plt.xticks([1,1.5,2,2.5,3], ['Tropical', 'Temporal', 'Boreal', 'Arid', 'Global'])
ax.set_axis_off()

figToPath = current_dir+'/4_Figures/Fig03b_CE_by_koppen_landsat_extreme_v2'
# fig.tight_layout()
# fig.savefig(figToPath, dpi=600)
# plt.close(fig)

fig, ax = plt.subplots(1,3,figsize=(6.5, 2.))
ax[0].hist(merged_df['CEts_tree'],20,ec='black')
ax[1].hist(merged_df['CEts_grass'],20,ec='black')
ax[2].hist(merged_df['CEts_cropland'],20,ec='black')
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03b_CE_hist_landsat_extreme'
fig.tight_layout()
# fig.savefig(figToPath, dpi=600)
# plt.close(fig)