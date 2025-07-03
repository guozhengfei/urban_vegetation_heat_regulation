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

def divide_2_class(array, threshold = 0.1):
    array[array<threshold] = 0
    array[array>threshold] = 1
    return array

def remove_outliers_replace_mean(arr, threshold=3):
    # Compute the mean and standard deviation of the array
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)

    # Define the upper and lower bounds for outliers
    lower_bound = arr_mean - threshold * arr_std
    upper_bound = arr_mean + threshold * arr_std

    # Replace outliers with the mean value
    cleaned_arr = np.where(np.logical_or(arr < lower_bound, arr > upper_bound), arr_mean, arr)

    return cleaned_arr

def smooth_2d_array(arr, window_size):
    kernel = np.ones(window_size) / window_size
    smoothed_arr = convolve1d(arr, kernel, axis=0, mode='nearest')
    return smoothed_arr

def gaussian_func(X, t0, a1, a2):
    x, y = X
    return t0 + a1 * x + a2 * y

def background_climate(tairHot,Dem,urban):

    # Generate some sample data points
    x = np.linspace(1,tairHot.shape[1],tairHot.shape[1])
    y = np.linspace(1,tairHot.shape[0],tairHot.shape[0])
    X, Y = np.meshgrid(x, y)
    Z = tairHot*1
    Dem_mean = np.nanmean(Dem*urban)
    # plt.figure();plt.imshow(Dem)
    Dem_std = np.nanstd(Dem*urban)
    dem = Dem*1.0
    dem_mask = (dem>(Dem_mean+3*Dem_std)) | (dem < (Dem_mean-3*Dem_std))
    dem[dem_mask] = np.nan
    # plt.figure(); plt.imshow(dem-Dem_mean)
    mask0 = np.isnan(dem) | np.isnan(Z)
    try:
        slope = st.linregress(dem[~mask0], Z[~mask0]).slope
        rvalue = st.linregress(dem[~mask0], Z[~mask0]).rvalue
    except ValueError:
        slope = -0.0065
        rvalue = 0

    if abs(rvalue) < 0.5: slope = -0.0065
    Z = Z - (dem - Dem_mean) * slope
    # Perform the custom function fit
    mask = ~np.isnan(urban) | np.isnan(Z)
    X1 = X[~mask]
    Y1 = Y[~mask]
    Z1 = Z[~mask]
    p0 = [np.mean(Z1), 1, 1]  # Initial guess for the parameters
    fit_params, _ = curve_fit(gaussian_func, (X1, Y1), Z1, p0)
    Z1_fit = gaussian_func((X1, Y1), *fit_params)
    st.linregress(Z1_fit,Z1)
    Z_fit = gaussian_func((X, Y), *fit_params)
    return Z_fit

def find_pure_pix(Dem, urban, grassC, cropC, treeC):
    Dem_mean = np.nanmean(Dem * urban)
    # plt.figure();plt.imshow(Dem)
    Dem_std = np.nanstd(Dem * urban)
    dem = Dem * 1.0
    dem_mask = (dem > (Dem_mean + 3 * Dem_std)) | (dem < (Dem_mean - 3 * Dem_std))
    dem[dem_mask] = np.nan
    grassD = np.sort(grassC[~np.isnan(dem)])[-15]
    cropD = np.sort(cropC[~np.isnan(dem)])[-15]
    treeD = np.sort(treeC[~np.isnan(dem)])[-15]

    mask = (grassC > grassD) | (treeC > treeD) | (cropC > cropD)
    urban2 = ~np.isnan(urban) | (mask & ~dem_mask)
    urban2 = urban2.astype(float)
    urban2[urban2 == 0] = np.nan
    return urban2, dem

## main ##
cities_ID = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\treeCover_100m\\'

grassC_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\grassCover_100m\\'

cropC_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\cropCover_100m\\'

# shrubC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\shrub_cover\\'

buildC_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\buildCover_100m\\'

# bareC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\bare_cover\\'

AlbedoTiff_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\Albedo\\'

urbanRaster_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\urbanRaster_100m\\'

# impervious_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\Impervious\\'

dem_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\DEM_100m\\'

# pop_folder = r'D:\Projects\Postdoc urban greening\Data\Population_density\\'

water_folder = r'D:\Projects\Postdoc urban greening\Landsat_scale_data_codes\waterCover_100m\\'

# hotMonthes = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\hotMonth\hotMonth_city.csv')

# hotMonthes['hotmonth']=np.round(hotMonthes[hotMonthes.columns[1:]].mean(axis=1))

CE = []
for i in cities_ID[1:]:
    if i in [703, 267, 443]: continue
    try:
        Albedo = tf.imread(AlbedoTiff_folder + 'albedo_' + str(i) + '.0.tif')
    except FileNotFoundError:
        continue
    # if np.isnan(np.nanmean(Albedo)): continue
    tairHot = Albedo
    # tairHot = smooth_2d_array(tairHot, window_size=7)
    # plt.figure(); plt.imshow(tairHot)
    treeC = tf.imread(treeC_folder + 'cropC_' + str(i) + '.0.tif') # name is wrong, values are correct
    # treeC = smooth_2d_array(treeC, window_size=7)
    grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') #+ tf.imread(
        #shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
    # grassC = smooth_2d_array(grassC, window_size=7)
    # shrubC = tf.imread(shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
    # shrubC = smooth_2d_array(shrubC, window_size=5)
    cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
    # cropC = smooth_2d_array(cropC, window_size=7)
    # bareC = tf.imread(bareC_folder + 'bareC_' + str(i) + '.0.tif')
    # bareC = smooth_2d_array(bareC, window_size=5)
    # buildC = tf.imread(impervious_folder + 'imperviousC_' + str(i) + '.0.tif')
    buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')
    # buildC = smooth_2d_array(buildC, window_size=7)
    waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
    # waterC = smooth_2d_array(waterC, window_size=7)
    # waterC[waterC>0.9]=np.nan

    Dem = tf.imread(dem_folder + 'dem_' + str(i) + '.0.tif')
    # pop = tf.imread(pop_folder + 'pop_' + str(i) + '.0.tif')[:,:,-1]

    urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan

    # tair_bg = background_climate(tairHot, Dem, urban)
    tair_bg = 0
    # urban, Dem = find_pure_pix(Dem, urban, grassC, cropC, treeC)

    # urban[0:5,:]=np.nan; urban[-5:,:]=np.nan; urban[:,0:5]=np.nan; urban[:,-5:]=np.nan;

    # plt.figure(); plt.imshow(Dem)
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
    if valid_ratio<0.2: continue
    # bareC = bareC * urban
    # pop = pop * urban

    # pop = pop.reshape(-1)
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
    X = np.array([tree[~mask], grass[~mask], crop[~mask], build[~mask], water[~mask], dem[~mask], tair[~mask]]).T  # ,
    X = pd.DataFrame(X)
    X.columns = ['tree', 'grass', 'crop', 'impervious', 'water', 'dem', 'alb']

    pure_tree = X.nlargest(50, 'tree')
    alb_tree = pure_tree.loc[pure_tree['tree']>0.95]['alb'].mean()

    pure_build = X.nlargest(50, 'impervious')
    alb_build = pure_build.loc[pure_build['impervious'] > 0.95]['alb'].mean()

    pure_grass = X.nlargest(50, 'grass')
    alb_grass = pure_grass.loc[pure_grass['grass'] > 0.95]['alb'].mean()

    pure_crop = X.nlargest(50, 'crop')
    alb_crop = pure_crop.loc[pure_crop['crop'] > 0.95]['alb'].mean()


    ce = [alb_tree, alb_grass, alb_crop, alb_build]
    CE.append([i] + ce)
    print(i)

CE = np.array(CE)
CE = pd.DataFrame(CE)
CE.columns = ['ID', 'alb_tree','alb_grass', 'alb_cropland', 'alb_build']
CE.to_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Alb_Landsat_yearly_v2.csv', index=False)
plt.figure(); plt.hist(IQR_filter(CE.iloc[:, -1]), 20, ec='black')

np.sum(CE.iloc[:, 1] > 0) / np.sum(~np.isnan(CE.iloc[:, 1]))
np.nanmean(CE, axis=0)