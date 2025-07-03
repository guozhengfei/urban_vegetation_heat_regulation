import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

CETs = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v5.csv')

climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

merged_df = pd.merge(CETs, climate, on='ID')

# tropical
merged_df[merged_df['koppen'] == 1]['CEts_tree'].mean()
merged_df[merged_df['koppen'] == 1]['CEts_grass'].mean()
merged_df[merged_df['koppen'] == 1]['CEts_cropland'].mean()

# arid
merged_df[merged_df['koppen']==2]['CEts_tree'].mean()
merged_df[merged_df['koppen']==2]['CEts_grass'].mean()
merged_df[merged_df['koppen']==2]['CEts_cropland'].mean()

# temporal
merged_df[merged_df['koppen']==3]['CEts_tree'].mean()
merged_df[merged_df['koppen']==3]['CEts_grass'].mean()
merged_df[merged_df['koppen']==3]['CEts_cropland'].mean()

# boreal
merged_df[merged_df['koppen']==4]['CEts_tree'].mean()
merged_df[merged_df['koppen']==4]['CEts_grass'].mean()
merged_df[merged_df['koppen']==4]['CEts_cropland'].mean()

fig, ax = plt.subplots(1,figsize=(4,6.5))
x = np.array([0.9, 1, 1.1])
ax.barh(x, merged_df[merged_df['koppen']==1].iloc[:,[1,3,5]].mean().values, xerr=merged_df[merged_df['koppen']==1].iloc[:,[1,3,5]].std(), height=0.1,color=['#7fc97f','#beaed4','#fdc086']) # tropical
ax.barh(x+0.5, merged_df[merged_df['koppen']==3].iloc[:,[1,3,5]].mean(), xerr=merged_df[merged_df['koppen']==3].iloc[:,[1,3,5]].std(), height=0.1,color=['#7fc97f','#beaed4','#fdc086']) # temporal
ax.barh(x+1, merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].mean(), xerr=merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].std(), height=0.1,color=['#7fc97f','#beaed4','#fdc086']) # boreal
ax.barh(x+1.5, merged_df[merged_df['koppen']==2].iloc[:,[1,3,5]].mean()+0.2, xerr=merged_df[merged_df['koppen']==2].iloc[:,[1,3,5]].std(), height=0.1,color=['#7fc97f','#beaed4','#fdc086']) # arid
ax.set_yticks([1,1.5,2,2.5], ['Tropical', 'Temporal', 'Boreal', 'Arid'])
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig01b_CE_by_koppen2'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

fig, ax = plt.subplots(1,3,figsize=(6.5, 2.))
ax[0].hist(merged_df['CEts_tree'],20,ec='black')
ax[1].hist(merged_df['CEts_grass'],20,ec='black')
ax[2].hist(merged_df['CEts_cropland'],20,ec='black')
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig01b_CE_hist'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)