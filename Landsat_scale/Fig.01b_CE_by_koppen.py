import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=10)
import matplotlib; #matplotlib.use('Qt5Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
from matplotlib.pyplot import MultipleLocator

current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\','/')
relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv'
# relative_path = '/2_Output/Cooling_Efficiency/CE_three_datasets.csv'
CETs = pd.read_csv(current_dir + relative_path)

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

merged_df = pd.merge(CETs, climate, on='ID')

fig, ax = plt.subplots(1,figsize=(6.5*0.7,3*0.7))
x = np.array([0.9, 1, 1.1])
ax.bar(x, merged_df[merged_df['koppen']==1].iloc[:,[1,3,5]].mean().values, yerr=merged_df[merged_df['koppen']==1].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # tropical
ax.bar(x+0.5, merged_df[merged_df['koppen']==3].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==3].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # temporal
ax.bar(x+1, merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # boreal
ax.bar(x+1.5, merged_df[merged_df['koppen']==2].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==2].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # arid
ax.set_xticks([1,1.5,2,2.5], ['Tropical', 'Temporal', 'Boreal', 'Arid'])
ax.tick_params(labelsize=10)

figToPath = current_dir + '/4_Figures/Fig01b_CE_by_koppen_landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

fig, ax = plt.subplots(1,3,figsize=(6., 1.8))
ax[0].hist(merged_df['CEts_tree'],20,ec='black')
ax[1].hist(merged_df['CEts_grass'],20,ec='black')
ax[2].hist(merged_df['CEts_cropland'],20,ec='black')
ax[0].tick_params(labelsize=12)
ax[1].tick_params(labelsize=12)
ax[2].tick_params(labelsize=12)
figToPath = current_dir + '/4_Figures/Fig01b_CE_hist_landsat_v3'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)


## CE based on Ta data
relative_path = '/2_Output/Cooling_Efficiency/CE_2020_yearly_v4.csv'
CETs = pd.read_csv(current_dir + relative_path)

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

merged_df = pd.merge(CETs, climate, on='ID')

fig, ax = plt.subplots(1,figsize=(6.5*0.7,3*0.7))
x = np.array([0.9, 1, 1.1])
ax.bar(x, merged_df[merged_df['koppen']==1].iloc[:,[1,3,5]].mean().values-0.1, yerr=merged_df[merged_df['koppen']==1].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # tropical
ax.bar(x+0.5, merged_df[merged_df['koppen']==3].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==3].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # temporal
ax.bar(x+1, merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # boreal
ax.bar(x+1.5, merged_df[merged_df['koppen']==2].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==2].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # arid
y_major_locator = MultipleLocator(0.3)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_xticks([1,1.5,2,2.5], ['Tropical', 'Temperate', 'Boreal', 'Arid'])
ax.tick_params(labelsize=10)

figToPath = current_dir + '/4_Figures/Fig01b_CE_by_koppen_Ta'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

