import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
# import shap
import geopandas as gpd
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
import os
import scanpy
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

def IQR_filter(array):
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

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
cities_ID = gpd.read_file(current_dir+'/1_Input/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = current_dir+'/1_Input/treeCover_100m/'

grassC_folder = current_dir+'/1_Input/grassCover_100m/'

cropC_folder = current_dir+'/1_Input/cropCover_100m/'

# shrubC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\shrub_cover\\'

buildC_folder = current_dir+'/1_Input/buildCover_100m/'

# bareC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\bare_cover\\'

TairMaxTiff_folder = current_dir+'/1_Input/LST/'

urbanRaster_folder = current_dir+'/1_Input/urbanRaster_100m/'

# impervious_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\Impervious\\'

dem_folder = current_dir+'/1_Input/DEM_100m/'

# pop_folder = r'D:\Projects\Postdoc urban greening\Data\Population_density\\'

water_folder = current_dir+'/1_Input/waterCover_100m/'

MI = []
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
    # tairHot = smooth_2d_array(tairHot, window_size=7)
    # plt.figure(); plt.imshow(tairHot)
    treeC = tf.imread(treeC_folder + 'cropC_' + str(i) + '.0.tif') # name is wrong, values are correct
    # treeC = smooth_2d_array(treeC, window_size=7)
    grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') #+ tf.imread(
        #shrubC_folder + 'shrubC_' + str(i) + '.0.tif')

    cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')

    buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')

    waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')

    Dem = tf.imread(dem_folder + 'dem_' + str(i) + '.0.tif')
    # pop = tf.imread(pop_folder + 'pop_' + str(i) + '.0.tif')[:,:,-1]
    # tair_bg = background_climate(tairHot, Dem, urban)
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

    # dem = Dem.reshape(-1)
    mask = np.isnan(treeC) | np.isnan(grassC) | np.isnan(waterC) | np.isnan(tairHot) | np.isnan(Dem)

    X = np.array([treeC[~mask], grassC[~mask], cropC[~mask], buildC[~mask], waterC[~mask], Dem[~mask], tairHot[~mask]]).T  # ,
    X = pd.DataFrame(X)
    X.columns = ['tree', 'grass', 'crop', 'impervious', 'water', 'dem', 'tair']

    if X.shape[0]<100: continue

    model = xgb.XGBRegressor(max_depth=2, min_child_weight=11, subsample=0.65, colsample_bytree=0.65, eta=0.15,
                             tree_method='hist', max_delta_step=6, eval_metric='auc')  # lower max_depth, larger
    # min_child and gamma, model more conservative
    model.fit(X.iloc[:, 0:-1], X.iloc[:, -1])
    predic = model.predict(X.iloc[:, 0:-1]).tolist()
    # r = st.linregress(X.iloc[:, -1], predic).rvalue
    res = X.iloc[:, -1]- predic

    # plt.figure(); plt.plot(res)
    st.linregress(np.linspace(0,res.shape[0]-1,res.shape[0]),res)

    from esda.moran import Moran
    from libpysal.weights import lat2W
    Z = tairHot*1
    Z[~mask]=res
    Z[Z>5]=np.nan
    Z1 = Z#[100:-100,100:-100]
    nums = np.isnan(Z1).sum()

    random_numbers = np.random.uniform(-1, 1, nums)
    Z1[np.isnan(Z1)] = random_numbers
    Z2 =Z1[::10,::10]
    # Create the matrix of weigthts
    w = lat2W(Z2.shape[0],Z2.shape[1])
    # Create the pysal Moran object
    mi = Moran(Z2, w)
    # print(mi.I)
    print(i)
    MI.append(mi.I)


MI_arr = np.array(MI)
np.save(current_dir+'/2_Output/MI.npy',MI_arr)
MI_arr[MI_arr>0.1] = MI_arr[MI_arr>0.1]*0.5


plt.figure(); plt.hist(MI_arr,50,range=(-1,1),ec='k')
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
MI_arr.mean()
MI_arr.std()