import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
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

cities_ID = gpd.read_file(current_dir + '/1_Input/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = current_dir + '/1_Input/Extreme_machanism_data/tree_cover_from10m/'

grassC_folder = current_dir + '/1_Input/Extreme_machanism_data/grass_cover_from10m/'

cropC_folder = current_dir + '/1_Input/Extreme_machanism_data/crop_cover_from10m/'

# buildC_folder = current_dir + '/1_Input/buildCover_100m/'

LEMaxTiff_folder_extrem = current_dir + '/1_Input/Extreme_machanism_data/LE_series1000m/'

urbanRaster_folder = current_dir + '/1_Input/urbanRaster_100m/'

water_folder = current_dir + '/1_Input/waterCover_100m/'

tree_xy = []
grass_xy = []
crop_xy = []
for i in cities_ID[1:]:
    if i in [703, 689]: continue
    urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan
    urban = urban[::10, ::10]

    try:
        LE = tf.imread(LEMaxTiff_folder_extrem + 'LE_' + str(i) + '.0.tif')
    except FileNotFoundError:
        continue
    LE_mean = np.nanmean(LE, axis=2)
    row_dif = urban.shape[0] - LE_mean.shape[0]
    col_dif = urban.shape[1] - LE_mean.shape[1]
    urban = np.concatenate((urban[:-3, :], urban[-3 + row_dif:, :]), axis=0)
    urban = np.concatenate((urban[:, :-3], urban[:, -3 + col_dif:]), axis=1)
    # plt.figure(); plt.imshow(urban)

    treeC = tf.imread(treeC_folder + 'treeC_' + str(i) + '.0.tif')
    treeC[np.isnan(urban)] = -1
    treeC[np.isnan(LE_mean)] = -1
    # Flatten the array and get the indices of the top 5 elements
    indices = np.argpartition(treeC.flatten(), -5)[-5:]
    top_5_indices = np.unravel_index(indices, treeC.shape)
    src = rasterio.open(treeC_folder + 'treeC_' + str(i) + '.0.tif')
    # return the coordinations of end-members
    x, y = src.xy(top_5_indices[0], top_5_indices[1])
    coord = np.array((x, y, 5 * [i])).T
    tree_xy.append(coord)

    grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif')
    grassC[np.isnan(urban)] = -1
    grassC[np.isnan(LE_mean)] = -1
    # plt.figure();plt.imshow(grassC)
    indices = np.argpartition(grassC.flatten(), -5)[-5:]
    top_5_indices = np.unravel_index(indices, grassC.shape)
    src = rasterio.open(grassC_folder + 'grassC_' + str(i) + '.0.tif')
    # return the coordinations of end-members
    x, y = src.xy(top_5_indices[0], top_5_indices[1])
    coord = np.array((x, y, 5 * [i])).T
    grass_xy.append(coord)

    cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
    cropC[np.isnan(urban)] = -1
    cropC[np.isnan(LE_mean)] = -1
    # plt.figure();plt.imshow(cropC)
    indices = np.argpartition(cropC.flatten(), -5)[-5:]
    top_5_indices = np.unravel_index(indices, cropC.shape)
    src = rasterio.open(cropC_folder + 'cropC_' + str(i) + '.0.tif')
    # return the coordinations of end-members
    x, y = src.xy(top_5_indices[0], top_5_indices[1])
    coord = np.array((x, y, 5 * [i])).T
    crop_xy.append(coord)
    print(i)

tree_xy = np.array(tree_xy)
tree_xy_rsp = np.reshape(tree_xy, (tree_xy.shape[0] * tree_xy.shape[1], tree_xy.shape[2]))

grass_xy = np.array(grass_xy)
grass_xy_rsp = np.reshape(grass_xy, (grass_xy.shape[0] * grass_xy.shape[1], grass_xy.shape[2]))

crop_xy = np.array(crop_xy)
crop_xy_rsp = np.reshape(crop_xy, (crop_xy.shape[0] * crop_xy.shape[1], crop_xy.shape[2]))

import ee

# Authenticate to the Earth Engine servers
ee.Authenticate()
ee.Initialize()

# VPD tempeseries
imageC = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').select("vpd").filterDate('2015-01-01', '2023-01-01').toBands();

VPD_values = []
ID_values = []
for i in cities_ID[1:]:
    if i in [703, 689]: continue
    src = rasterio.open(urbanRaster_folder + 'urban_' + str(i) + '.0.tif')
    bounds = src.bounds
    # Calculate the central coordinates
    central_x = (bounds.left + bounds.right) / 2
    central_y = (bounds.top + bounds.bottom) / 2
    point = ee.Geometry.Point(central_x, central_y)
    band_values = imageC.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10000)
    values_dict = band_values.getInfo()
    values = list(values_dict.values())
    VPD_values.append(values)
    id_values = [i] * len(values)
    ID_values.append(id_values)
    print(i)

VPD_array = np.array(VPD_values)
np.save(current_dir + '/2_Output/Cooling_Efficiency/VPD_array.npy', VPD_array)

names_VPD = list(values_dict.keys())
ID_values2 = list(cities_ID[1:].values)
ID_values2.remove(689);
ID_values2.remove(703);
ID_values3 = np.array(ID_values2).reshape(1, -1)

VPD_array = np.load(current_dir + '/2_Output/Cooling_Efficiency/VPD_array.npy', allow_pickle=True)
VPD_array2 = np.concatenate((VPD_array, ID_values3.T), axis=1)
names_VPD2 = names_VPD + ['ID']

df_VPD = pd.DataFrame(VPD_array2)
df_VPD.columns = names_VPD2
df_VPD.to_csv(current_dir + '/2_Output/Cooling_Efficiency/VPD_series.csv', index=False)