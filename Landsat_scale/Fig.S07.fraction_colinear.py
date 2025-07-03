import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import geopandas as gpd
from scipy.ndimage import convolve1d
import os
import seaborn as sns

def IQR_filter(array):
    p25 = np.percentile(array[~np.isnan(array)], 25)
    p75 = np.percentile(array[~np.isnan(array)],75)
    IQR = p75-p25
    array[array < p25 - 1.5*IQR] = np.nan
    array[array > p75 + 1.5*IQR] = np.nan
    arraynew = array
    return arraynew

def smooth_2d_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve1d(arr, kernel, axis=0, mode='nearest')
    return smoothed_arr

## main ##
current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

cities_ID = gpd.read_file(current_dir + '/1_Input/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = current_dir + '/1_Input/treeCover_100m/'

grassC_folder = current_dir + '/1_Input/grassCover_100m/'

cropC_folder = current_dir + '/1_Input/cropCover_100m/'

buildC_folder = current_dir + '/1_Input/buildCover_100m/'

TairMaxTiff_folder_extrem = current_dir + '/1_Input/LST_extreme/85th/'

TairMaxTiff_folder_normal = current_dir + '/1_Input/LST/'

urbanRaster_folder = current_dir + '/1_Input/urbanRaster_100m/'


dem_folder = current_dir + '/1_Input/DEM_100m/'

water_folder = current_dir + '/1_Input/waterCover_100m/'

# hotMonthes['hotmonth']=np.round(hotMonthes[hotMonthes.columns[1:]].mean(axis=1))

i=6
urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
urban[urban != i] = 0
urban[urban == i] = 1
urban[urban == 0] = np.nan

tairMax_extreme = tf.imread(TairMaxTiff_folder_extrem + 'Landsat_' + str(i) + '.0.tif')

tairHot = tairMax_extreme
valid_ratio = np.sum(~np.isnan(tairHot*urban)) / np.sum(~np.isnan(urban))
# tairHot = smooth_2d_array(tairHot, window_size=7)
plt.figure(); plt.imshow(tairHot*urban,cmap='RdYlBu_r',vmin=40,vmax=50)
treeC = tf.imread(treeC_folder + 'cropC_' + str(i) + '.0.tif') # name is wrong, values are correct
# treeC = smooth_2d_array(treeC, window_size=7)
plt.figure(); plt.imshow(treeC*urban)

grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') #+ tf.imread(
    #shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
# grassC = smooth_2d_array(grassC, window_size=7)
plt.figure(); plt.imshow(grassC*urban)

cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
# cropC = smooth_2d_array(cropC, window_size=7)
plt.figure(); plt.imshow(cropC*urban)

buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')
# buildC = smooth_2d_array(buildC, window_size=7)
plt.figure(); plt.imshow(buildC*urban)

waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
# waterC = smooth_2d_array(waterC, window_size=7)
# waterC[waterC>0.9]=np.nan
plt.figure(); plt.imshow(waterC*urban)

Dem = tf.imread(dem_folder + 'dem_' + str(i) + '.0.tif')
plt.figure(); plt.imshow(Dem*urban)

waterC = waterC * urban
treeC = treeC * urban
grassC = grassC * urban
cropC = cropC * urban
buildC = buildC * urban
tairHot = tairHot * urban

tree = treeC.reshape(-1)
grass = grassC.reshape(-1)
crop = cropC.reshape(-1)
build = buildC.reshape(-1)
water = waterC.reshape(-1)
tair = tairHot.reshape(-1)

dem = Dem.reshape(-1)
mask = np.isnan(tree) | np.isnan(grass) | np.isnan(water) | np.isnan(tair) | np.isnan(dem)

# plt.figure(); plt.hist(tree[~mask], bins = 50)
# Prepare the data
X = np.array([tree[~mask], grass[~mask], crop[~mask], build[~mask], water[~mask], dem[~mask]]).T  # ,
X = pd.DataFrame(X)
X.columns = ['tree', 'grass', 'crop', 'built-up', 'water', 'dem']

corr = X.corr()
corr = corr*1.5; corr[corr>1]=1
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
