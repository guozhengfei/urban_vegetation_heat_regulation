import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
import matplotlib;

matplotlib.use('Qt5Agg')
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
current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')

df_ET_tree = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ET_tree.csv')
df_ET_grass = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ET_grass.csv')
df_ET_crop = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ET_crop.csv')

df_ET_tree = df_ET_tree.drop(columns=['lon', 'lat'])
df_ET_grass = df_ET_grass.drop(columns=['lon', 'lat'])
df_ET_crop = df_ET_crop.drop(columns=['lon', 'lat'])

df_ET_tree2 = df_ET_tree.groupby(['ID'], as_index=False).agg('mean')
df_ET_grass2 = df_ET_grass.groupby(['ID'], as_index=False).agg('mean')
df_ET_crop2 = df_ET_crop.groupby(['ID'], as_index=False).agg('mean')

df_green_tree = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Greeness_tree.csv')
df_green_grass = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Greeness_grass.csv')
df_green_crop = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Greeness_crop.csv')

df_green_tree = df_green_tree.drop(columns=['lon', 'lat'])
df_green_grass = df_green_grass.drop(columns=['lon', 'lat'])
df_green_crop = df_green_crop.drop(columns=['lon', 'lat'])

df_green_tree2 = df_green_tree.groupby(['ID'], as_index=False).agg('mean')
df_green_grass2 = df_green_grass.groupby(['ID'], as_index=False).agg('mean')
df_green_crop2 = df_green_crop.groupby(['ID'], as_index=False).agg('mean')

world = gpd.read_file(current_dir + '/2_Output/Shp/world_countries.shp')
cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')

# the countries numbers including at least one point
points_in_polygons = gpd.sjoin(cities, world, how="left", op='intersects')
num_polygons_with_points = points_in_polygons['index_right'].nunique()
print(f"Number of polygons with at least one point inside: {num_polygons_with_points}")

# tropical = gpd.read_file(current_dir + '/2_Output/Shp/koppen_tropical.shp')
# arid = gpd.read_file(current_dir + '/2_Output/Shp/koppen_arid.shp')
# temperate = gpd.read_file(current_dir + '/2_Output/Shp/koppen_temperate.shp')
# boreal = gpd.read_file(current_dir + '/2_Output/Shp/koppen_boreal.shp')

# # the points within the polygon
# points_in_polygons = gpd.sjoin(cities, temperate, how="inner", op='within')
# points_within_temperate = points_in_polygons[['ID','geometry']]
# points_within_temperate = points_within_temperate[points_within_temperate.geometry.y > -15]

cities_extra_tropic = cities[cities.geometry.y > 15]
ET_tree_extropic = pd.merge(cities_extra_tropic,df_ET_tree2,on='ID')
ET_tree_mean = np.nanmean(ET_tree_extropic.iloc[:,3:].values,axis=0)
ET_tree = np.nanmean(ET_tree_mean.reshape(8,46),axis=0)*0.1

ET_grass_extropic = pd.merge(cities_extra_tropic,df_ET_grass2,on='ID')
ET_grass_mean = np.nanmean(ET_grass_extropic.iloc[:,3:].values,axis=0)
ET_grass = np.nanmean(ET_grass_mean.reshape(8,46),axis=0)*0.1

ET_crop_extropic = pd.merge(cities_extra_tropic,df_ET_crop2,on='ID')
ET_crop_mean = np.nanmean(ET_crop_extropic.iloc[:,3:].values,axis=0)
ET_crop = np.nanmean(ET_crop_mean.reshape(8,46),axis=0)*0.1

green_tree_extropic = pd.merge(cities_extra_tropic,df_green_tree2,on='ID')
green_tree_mean = np.nanmean(green_tree_extropic.iloc[:,3:].values,axis=0)
green_tree = np.nanmean(green_tree_mean.reshape(8,12),axis=0)*0.0001

green_grass_extropic = pd.merge(cities_extra_tropic,df_green_grass2,on='ID')
green_grass_mean = np.nanmean(green_grass_extropic.iloc[:,3:].values,axis=0)
green_grass = np.nanmean(green_grass_mean.reshape(8,12),axis=0)*0.0001

green_crop_extropic = pd.merge(cities_extra_tropic,df_green_crop2,on='ID')
green_crop_mean = np.nanmean(green_crop_extropic.iloc[:,3:].values,axis=0)
green_crop = np.nanmean(green_crop_mean.reshape(8,12),axis=0)*0.0001


fig, ax = plt.subplots(1, 3, figsize=(9, 2))
ax[0].plot(ET_tree[::4],lw=2)
ax[0].tick_params(axis='y', colors='C0')
ax[1].plot(ET_grass[::4],lw=2)
ax[1].tick_params(axis='y', colors='C0')
ax[2].plot(ET_crop[::4],lw=2)
ax[2].tick_params(axis='y', colors='C0')

ax10 = ax[0].twinx()
ax10.plot(green_tree,'C1-',lw=2)
ax10.tick_params(axis='y', colors='C1')
ax11 = ax[1].twinx()
ax11.plot(green_grass,'C1-',lw=2)
ax11.tick_params(axis='y', colors='C1')
ax12 = ax[2].twinx()
ax12.plot(green_crop,'C1-',lw=2)
ax12.tick_params(axis='y', colors='C1')
fig.tight_layout()
ax[0].set_xticks([0,3,6,9], ['Winter', 'Spring', 'Summer', 'Autumn'])
ax[1].set_xticks([0,3,6,9], ['Winter', 'Spring', 'Summer', 'Autumn'])
ax[2].set_xticks([0,3,6,9], ['Winter', 'Spring', 'Summer', 'Autumn'])

figToPath = current_dir + '/4_Figures/Fig08_urban_ET_greenness_seasonality'
fig.savefig(figToPath, dpi=600)
# plt.close(fig)
