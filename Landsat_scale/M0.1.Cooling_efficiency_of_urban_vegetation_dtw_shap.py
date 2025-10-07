import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
import os
import geopandas as gpd
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
import shap

# Perform linear regression along the 3rd dimension
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

def gaussian_func(X, t0, a1, a2):
    x, y = X
    return t0 + a1 * x + a2 * y

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
dtw_folder = current_dir + '/1_Input/Distance_to_water/'

dem_folder = current_dir + '/1_Input/DEM_100m/'
water_folder = current_dir + '/1_Input/waterCover_100m/'


CE = []
for i in cities_ID[1:]:
    if i in [703,689]: continue
    urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan

    try:
        tairMax = tf.imread(TairMaxTiff_folder + 'Landsat_' + str(i) + '.0.tif')
    except FileNotFoundError:
        continue
    valid_ratio = np.sum(~np.isnan(np.nanmean(tairMax,axis=2)*urban)) / np.sum(~np.isnan(urban))
    # plt.figure(); plt.imshow(urban)
    if valid_ratio < 0.6: continue
    # gap-fill
    tairMax_fill = gap_fill_tair(tairMax, urban)
    tairHot = IQR_filter2(np.nanmean(tairMax_fill, axis=2))


    tairHot[np.isnan(tairHot)] = np.nanmean(tairHot)
    treeC = tf.imread(treeC_folder + 'cropC_' + str(i) + '.0.tif') # name is wrong, values are correct
    grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') #+ tf.imread(
    cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
    buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')
    waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
    dtwC = tf.imread(dtw_folder+'dtw_'+ str(i) + '.0.tif')

    Dem = tf.imread(dem_folder + 'dem_' + str(i) + '.0.tif')
    tair_bg = 0

    Dem = Dem*urban
    Dem0 = Dem * 1
    Dem[Dem > (np.nanmean(Dem0) + 3 * np.nanstd(Dem0))] = np.nan
    Dem[Dem < (np.nanmean(Dem0) - 3 * np.nanstd(Dem0))] = np.nan
    waterC = waterC * urban
    treeC = treeC * urban
    grassC = grassC * urban
    cropC = cropC * urban
    buildC = buildC * urban
    tairHot = tairHot * urban
    dtwC = dtwC*urban

    tree = treeC.reshape(-1)
    grass = grassC.reshape(-1)
    crop = cropC.reshape(-1)
    build = buildC.reshape(-1)
    water = waterC.reshape(-1)
    tair = (tairHot - tair_bg).reshape(-1)
    dtw = dtwC.reshape(-1)

    tair[tair>np.nanmean(tair)+3*np.nanstd(tair)]=np.nan
    tair[tair<np.nanmean(tair)-3*np.nanstd(tair)]=np.nan

    dem = Dem.reshape(-1)
    mask = np.isnan(tree) | np.isnan(grass) | np.isnan(water) | np.isnan(tair) | np.isnan(dem)

    X = np.array([tree[~mask], grass[~mask], crop[~mask], build[~mask], water[~mask], dem[~mask], dtw[~mask], tair[~mask]]).T  # ,
    X = pd.DataFrame(X)
    X.columns = ['tree', 'grass', 'crop', 'impervious', 'water', 'dem', 'dtw', 'tair']

    if X.shape[0]<100: continue
    factor = int(X.shape[0]/10000)
    if factor<1: factor=1
    X=X.iloc[::int(factor),:]

    model = xgb.XGBRegressor(max_depth=2, min_child_weight=11, subsample=0.65, colsample_bytree=0.65, eta=0.15,
                             tree_method='hist', max_delta_step=6, eval_metric='auc')  # lower max_depth, larger
    # min_child and gamma, model more conservative
    model.fit(X.iloc[:, 0:-1], X.iloc[:, -1])

    # Calculate SHAP values
    explainer = shap.Explainer(model, X.iloc[:, 0:-1])
    shap_values = explainer(X.iloc[:, 0:-1])
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

    predic = model.predict(X.iloc[:, 0:-1]).tolist()
    r = st.linregress(X.iloc[:, -1], predic).rvalue

    # ce = mean_slope(data,values)
    shap_list = list(mean_abs_shap)
    CE.append([i] + shap_list)
    print(i)

CE = np.array(CE)
CE = pd.DataFrame(CE)
CE.columns = ['ID', 'shap_tree', 'shap_grass', 'shap_crop', 'shap_impervious', 'shap_water', 'shap_dem', 'shap_dtw']
CE.to_csv(current_dir+'/2_Output/Cooling_Efficiency/shap_Landsat_yearly_Ts_dtw_v2.1.csv', index=False)
