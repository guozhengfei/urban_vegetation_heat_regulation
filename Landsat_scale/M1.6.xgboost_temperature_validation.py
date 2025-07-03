import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; #matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
# import shap
import geopandas as gpd
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
import os

# Perform linear regression along the 3rd dimension
def linear_regression_3d_with_nan(array):
    # Reshape the array to 2D (flatten along the 3rd dimension)
    flattened_array = (array*1).reshape((-1, array.shape[2]))

    # Find indices of NaN values
    nan_indices = np.isnan(flattened_array)

    # Replace NaN values with the mean of non-NaN values along each column
    column_means = np.nanmean(flattened_array, axis=0)
    flattened_array[nan_indices] = np.take(column_means, np.where(nan_indices)[1])

    # Perform linear regression
    x = np.arange(flattened_array.shape[1])
    slope, intercept = np.polyfit(x, flattened_array.T, deg=1)
    flattened_array = array.reshape((-1, array.shape[2]))
    slope = (slope+flattened_array[:,1]-flattened_array[:,1]).reshape(array.shape[0], array.shape[1])
    intercept = (intercept++flattened_array[:,1]-flattened_array[:,1]).reshape(array.shape[0], array.shape[1])
    return slope, intercept


    p25 = np.percentile(array[~np.isnan(array)], 25)
    p75 = np.percentile(array[~np.isnan(array)],75)
    IQR = p75-p25
    array[array < p25 - 1.5*IQR] = np.nan
    array[array > p75 + 1.5*IQR] = np.nan
    arraynew = array
    return arraynew

def IQR_filter2(array):
    std = np.nanstd(array)
    mean = np.nanmean(array)
    array[array < mean - 3*std] = np.nan
    array[array > mean + 3*std] = np.nan
    arraynew = array
    return arraynew

def smooth_2d_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve1d(arr, kernel, axis=0, mode='nearest')
    return smoothed_arr

def gap_fill_tair(tairMax, urban):  # only consider the tair within urban
    # make urban have the same size with tair data
    repeated_urban = np.repeat(urban[:, :, np.newaxis], tairMax.shape[-1], axis=2)
    tairMax = tairMax * repeated_urban

    # reshape the 3d array to 2d: remove the data outside the urban
    tairMax_2d = tairMax.reshape((-1, tairMax.shape[-1]))
    urban_2d = repeated_urban.reshape((-1, repeated_urban.shape[-1]))
    tairMax_urban = tairMax_2d[~np.isnan(urban_2d[:, 1]), :]

    # reshape the 2d array to 1d for linear interpolate
    tairMax_urban_1d = tairMax_urban.reshape(-1)
    mask = np.isnan(tairMax_urban_1d)

    # Get the indices of non-NaN values
    indices = np.arange(len(tairMax_urban_1d))

    # Perform linear interpolation
    tairMax_urban_1d[mask] = np.interp(indices[mask], indices[~mask], tairMax_urban_1d[~mask])

    tairMax_urban_fill = tairMax_urban_1d.reshape(tairMax_urban.shape)
    # reshape the fill array to original size
    tairMax_2d[~np.isnan(urban_2d[:, 1]), :] = tairMax_urban_fill
    tairMax_fill = tairMax_2d.reshape(tairMax.shape)
    return tairMax_fill

## main ##
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\', '/')
cities_ID = gpd.read_file(current_dir + '/1_Input/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = current_dir + '/1_Input/treeCover_100m/'

grassC_folder = current_dir + '/1_Input/grassCover_100m/'

cropC_folder = current_dir + '/1_Input/cropCover_100m/'

buildC_folder = current_dir + '/1_Input/buildCover_100m/'

TairMaxTiff_folder = current_dir + '/1_Input/LST/'

urbanRaster_folder = current_dir + '/1_Input/urbanRaster_100m/'

dem_folder = current_dir + '/1_Input/DEM_100m/'

water_folder = current_dir + '/1_Input/waterCover_100m/'

CE = []
for i in cities_ID[1:]:
    if i in [703]: continue
    urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan

    try:
        tairMax = tf.imread(TairMaxTiff_folder + 'Landsat_' + str(i) + '.0.tif')
    except FileNotFoundError:
        continue

    # gap-fill
    tairMax_fill = gap_fill_tair(tairMax, urban)
    tairMax = tairMax_fill
    tairHot_mean = IQR_filter2(np.nanmean(tairMax_fill, axis=2))
    valid_ratio = np.sum(~np.isnan(tairHot_mean)) / np.sum(~np.isnan(urban))

    if valid_ratio < 0.2: continue
    for month in range(12):
        if tairMax.shape[2]-1 < month: month = tairMax.shape[2]-1
        tairHot = tairMax[:,:,month]
        tairHot = smooth_2d_array(tairHot, window_size=10)
        # plt.figure(); plt.imshow(tairHot)
        treeC = tf.imread(treeC_folder + 'cropC_' + str(i) + '.0.tif') # name is wrong, values are correct
        treeC = smooth_2d_array(treeC, window_size=10)
        grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') 
        grassC = smooth_2d_array(grassC, window_size=10)
        cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
        cropC = smooth_2d_array(cropC, window_size=10)
        buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')
        buildC = smooth_2d_array(buildC, window_size=10)
        waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
        waterC = smooth_2d_array(waterC, window_size=10)

        Dem = tf.imread(dem_folder + 'dem_' + str(i) + '.0.tif')

        urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
        urban[urban != i] = 0
        urban[urban == i] = 1
        urban[urban == 0] = np.nan

        tair_bg = 0

        Dem = Dem*urban
        Dem0 = Dem * 1
        Dem[Dem > (np.nanmean(Dem0) + 3 * np.nanstd(Dem0))] = np.nan
        Dem[Dem < (np.nanmean(Dem0) - 3 * np.nanstd(Dem0))] = np.nan
        waterC = waterC * urban
        treeC = treeC * urban
        grassC = grassC * urban
        # shrubC = shrubC * urban
        cropC = cropC * urban
        buildC = buildC * urban
        tairHot = tairHot * urban
        valid_ratio = np.sum(~np.isnan(tairHot)) / np.sum(~np.isnan(urban))
        if valid_ratio < 0.2: continue
        
        tree = treeC.reshape(-1)
        grass = grassC.reshape(-1)
        crop = cropC.reshape(-1)
        build = buildC.reshape(-1)
        water = waterC.reshape(-1)
        tair = (tairHot - tair_bg).reshape(-1)
        dem = Dem.reshape(-1)
        mask = np.isnan(tree) | np.isnan(grass) | np.isnan(water) | np.isnan(tair) | np.isnan(dem)

        # plt.figure(); plt.hist(tree[~mask], bins = 50)
        # Prepare the data
        X = np.array([tree[~mask], grass[~mask], crop[~mask], build[~mask],  tair[~mask]]).T  # ,
        X = pd.DataFrame(X)
        X.columns = ['tree_fra', 'grass_fra', 'crop_fra', 'impervious_fra', 'tair']
        # ts mean for tree_fra > 98%
        ts_tree = X.loc[X['tree_fra'] > 0.98, 'tair'].mean()
        ts_grass = X.loc[X['grass_fra'] > 0.98, 'tair'].mean()
        ts_crop = X.loc[X['crop_fra'] > 0.98, 'tair'].mean()
        ts_build = X.loc[X['impervious_fra'] > 0.98, 'tair'].mean()

        ce_tree = ts_tree - ts_build  # negative is cooling
        ce_grass = ts_grass - ts_build  # negative is cooling
        ce_crop = ts_crop - ts_build  # negative is cooling

        # ce = mean_slope(data,values)
        ce = [ce_tree, ts_tree, ce_grass, ts_grass, ce_crop, ts_crop, ts_build]
        CE.append([i] + [month] + ce)
    # CE.append([i] + ce)
    print(i)

CE = np.array(CE)
CE = pd.DataFrame(CE)
CE.columns = ['ID', 'month','CEts_tree_l','ts_tree_l','CEts_grass_l', 'ts_grass_l','CEts_cropland_l','ts_cropland_l','ts_build_l']
CE.to_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3_endmember.csv', index=False)

