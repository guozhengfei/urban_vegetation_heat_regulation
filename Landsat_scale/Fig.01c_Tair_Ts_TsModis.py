import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import matplotlib; #matplotlib.use('Qt5Agg')
import os
import numpy as np
import scipy.stats as st
from matplotlib.pyplot import MultipleLocator

current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\','/')
# relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv'
# CETs_L = pd.read_csv(current_dir+relative_path)[['ID','CEts_tree','CEts_grass','r']] # v5,v4.2
#
# relative_path = '/2_Output/Cooling_Efficiency/CE_2020_yearly_Ts_v4.3.csv'
# CETs_M = pd.read_csv(current_dir+relative_path)[['ID','CEts_tree','CEts_grass','r']]
#
# relative_path = '/2_Output/Cooling_Efficiency/CE_2020_yearly_v4.csv'
# CETa= pd.read_csv(current_dir+relative_path)[['ID','CEta_tree','CEta_grass','r']]
#
# df_merge = pd.merge(CETs_L,CETs_M,on='ID',how='left')
# df_merge = pd.merge(df_merge,CETa,on='ID',how='left')

relative_path = '/2_Output/Cooling_Efficiency/CE_three_datasets.csv'
df_merge = pd.read_csv(current_dir+relative_path)

x = df_merge['CEts_tree_y'].values
y1 = df_merge['CEts_tree_x'].values
y2 = df_merge['CEta_tree'].values
# diff = abs(y1-x)
# x[diff>5] = x[diff>5]-(x[diff>5]-y1[diff>5])*0.5
#
# diff2 = abs(y2-x)
# y2[diff2>8] = y2[diff2>8]-(y2[diff2>8]-x[diff2>8])*0.5
fig, axs = plt.subplots(2,1,figsize=(4.0*0.7, 6.5*0.7))
axs[0].plot(x, y1, 'o',color='#EC3E31',mfc="none",alpha=1)
axs[0].set_ylim(np.array([-10.0,4.0]))
axs[0].tick_params(axis='y', labelcolor='#EC3E31')
ax2 = axs[0].twinx()
ax2.plot(x, y2, '*',color='#4F99C9',mfc="none",alpha=0.9)
ax2.tick_params(axis='y', labelcolor='#4F99C9')
ax2.set_ylim(np.array([-2.7,1.8]))
coef = st.linregress(x,y1)
axs[0].plot([x.min(),x.max()],[coef.slope*x.min()+coef.intercept,coef.slope*x.max()+coef.intercept],'-',color='#EC3E31' )
coef = st.linregress(x[~np.isnan(y2)],y2[~np.isnan(y2)])
ax2.plot([x.min(),x.max()],[coef.slope*x.min()+coef.intercept,coef.slope*x.max()+coef.intercept],'-',color='#4F99C9' )
axs[0].tick_params(labelsize=12)
ax2.tick_params(labelsize=12)

x = df_merge['CEts_grass_y'].values
y1 = df_merge['CEts_grass_x'].values
y2 = df_merge['CEta_grass'].values
# diff = abs(y1-x)
# x[diff>4] = x[diff>4]-(x[diff>4]-y1[diff>4])*0.5
# diff2 = abs(y2*5-x)
# y2[diff2>4] = y2[diff2>4]-(y2[diff2>4]-x[diff2>4]/5)*0.5
axs[1].plot(x, y1, 'o',color='#EC3E31',mfc="none")
axs[1].tick_params(axis='y', labelcolor='#EC3E31')
axs[1].set_ylim(np.array([-9.0,7]))
ax3 = axs[1].twinx()
ax3.plot(x, y2, '*',color='#4F99C9',mfc="none",alpha=0.9)
ax3.tick_params(axis='y', labelcolor='#4F99C9')
ax3.set_ylim(np.array([-2,1.5]))
coef = st.linregress(x,y1)
axs[1].plot([x.min(),x.max()],[coef.slope*x.min()+coef.intercept,coef.slope*x.max()+coef.intercept],'-',color='#EC3E31' )
coef = st.linregress(x[~np.isnan(y2)],y2[~np.isnan(y2)])
ax3.plot([x.min(),x.max()],[coef.slope*x.min()+coef.intercept,coef.slope*x.max()+coef.intercept],'-',color='#4F99C9' )
axs[1].tick_params(labelsize=12)
ax3.tick_params(labelsize=12)
y_major_locator = MultipleLocator(1)
ax3.yaxis.set_major_locator(y_major_locator)

fig.tight_layout()

figToPath = current_dir + '/4_Figures/Fig01c'
fig.savefig(figToPath, dpi=600)
# plt.close(fig)