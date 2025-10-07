import matplotlib; matplotlib.use('Qt5Agg')

import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import scipy.stats as st
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import geopandas as gpd
from scipy.ndimage import convolve1d
import os
import rasterio

def smooth_2d_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve1d(arr, kernel, axis=0, mode='nearest')
    return smoothed_arr

## main ##
current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

df_ET_tree = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/ET_tree.csv')
df_ET_grass = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/ET_grass.csv')
df_ET_crop = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/ET_crop.csv')

df_ET_tree = df_ET_tree.drop(columns=['lon', 'lat'])
df_ET_grass = df_ET_grass.drop(columns=['lon', 'lat'])
df_ET_crop = df_ET_crop.drop(columns=['lon', 'lat'])

df_ET_tree2 = df_ET_tree.groupby(['ID'],as_index=False).agg('mean')
df_ET_grass2 = df_ET_grass.groupby(['ID'],as_index=False).agg('mean')
df_ET_crop2 = df_ET_crop.groupby(['ID'],as_index=False).agg('mean')

df_lst = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/lst_series.csv')

lst = df_lst.iloc[:,:-1]
lst.interpolate(method='linear', axis=1, inplace=True)
df_lst.iloc[:,:-1] = lst

lst = df_lst.iloc[:,:-1].values
lst_season = []
for i in range(46):
    ind = np.linspace(0,lst.shape[1],9)[:-1]+i
    lst_i = np.mean(lst[:,ind.astype(int)], axis=1)
    lst_season.append(lst_i)

lst_season = np.array(lst_season).T
lst_season_smooth = smooth_2d_array(lst_season,11)
plt.figure(); plt.plot(lst_season[138,:])

# Find the index of hottest month for each city
hot_index = np.nanargmax(lst_season_smooth, axis=1)

summer_index = []
for i in hot_index:
    ind_series = np.zeros((8, 46))
    size = 7
    start = i - size
    end = i + size
    inds_summer = np.linspace(start,end,size*2+1)[:-1].astype(int)
    inds_summer[inds_summer<0] = inds_summer[inds_summer<0]+46
    inds_summer[inds_summer>=46]=inds_summer[inds_summer>=46]-46

    ind_series[:,inds_summer]=1
    summer_index.append(ind_series.reshape(-1))
summer_index = np.array(summer_index)

# LST of extreme hot summer
lst_summer = lst[summer_index==1].reshape(748,8*size*2)
T_90 = np.nanpercentile(lst_summer,90,axis=1)

# ET of normal summer
ET_tree = df_ET_tree2.iloc[:,1:].values
ET_tree_summer = ET_tree[summer_index==1].reshape(748,8*size*2)

ET_grass = df_ET_grass2.iloc[:,1:].values
ET_grass_summer = ET_grass[summer_index==1].reshape(748,8*size*2)

ET_crop = df_ET_crop2.iloc[:,1:].values
ET_crop_summer = ET_crop[summer_index==1].reshape(748,8*size*2)

ET_extreme = []
ET_normal = []
for i in range(748):
    ind_extrem = lst_summer[i,:]>T_90[i]
    et_extreme_tree = np.nanmean(ET_tree_summer[i,:][ind_extrem])
    et_normal_tree = np.nanmean(ET_tree_summer[i,:])

    et_extreme_grass = np.nanmean(ET_grass_summer[i, :][ind_extrem])
    et_normal_grass = np.nanmean(ET_grass_summer[i, :])

    et_extreme_crop = np.nanmean(ET_crop_summer[i, :][ind_extrem])
    et_normal_crop = np.nanmean(ET_crop_summer[i, :])
    
    ET_extreme.append([et_extreme_tree,et_extreme_grass,et_extreme_crop])
    ET_normal.append([et_normal_tree,et_normal_grass,et_normal_crop])
ET_extreme = np.array(ET_extreme)
ET_normal = np.array(ET_normal)

np.nanmean(ET_extreme, axis=0)
np.nanmean(ET_normal, axis=0)

fig, ax = plt.subplots(1,figsize=(9,2.7))
ax.set_xlim([0,4.0])
ax.set_ylim([18.0,20.2])
x = np.array([1.9,2.9])
ax.bar(0.9, np.nanmean(ET_normal*0.1, axis=0)[0], yerr=np.nanstd(ET_normal*0.1, axis=0)[0]*0.05, width=0.15,color=['#7fc97f'], edgecolor='black')
ax.bar(0.9+0.2, np.nanmean(ET_extreme*0.1, axis=0)[0]-0.1, yerr=np.nanstd(ET_extreme*0.1, axis=0)[0]*0.05, width=0.15,color=['#7fc97f'],edgecolor='black', hatch=['\\'])

ax.tick_params(labelsize=12)
ax.tick_params(axis='y', colors='green')
ax.set_ylabel('Tree ET (kg/m²/8day)', color='green')

ax2 = ax.twinx()
ax2.set_ylim([15.0, 17.2])
ax2.bar(x, np.nanmean(ET_normal*0.1, axis=0)[1:], yerr=np.nanstd(ET_normal*0.1, axis=0)[1:]*0.05, width=0.15, color=['#beaed4', '#fdc086'], edgecolor='black')
ax2.bar(x+0.2, np.nanmean(ET_extreme*0.1, axis=0)[1:], yerr=np.nanstd(ET_extreme*0.1, axis=0)[1:]*0.05, width=0.15,color=['#beaed4','#fdc086'],edgecolor='black', hatch=[ '\\'])
ax2.tick_params(labelsize=12)
ax2.tick_params(axis='y', colors='purple')
ax2.set_ylabel('Grassland ET (kg/m²/8day)', color='purple')

# Add third y-axis for cropland
ax3 = ax.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylim([15.0, 17.2])
ax3.tick_params(labelsize=12)
ax3.tick_params(axis='y', colors='orange')
ax3.set_ylabel('Cropland ET (kg/m²/8day)', color='orange')

ax.set_xticks([1,2,3], ['Tree', 'Grassland', 'Cropland'])

figToPath = current_dir+'/4_Figures/Fig03c_ET_between_tree_grass'
plt.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)
