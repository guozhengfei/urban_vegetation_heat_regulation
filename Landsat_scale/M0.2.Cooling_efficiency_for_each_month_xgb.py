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

# impervious_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\Impervious\\'

dem_folder = current_dir + '/1_Input/DEM_100m/'

# pop_folder = r'D:\Projects\Postdoc urban greening\Data\Population_density\\'

water_folder = current_dir + '/1_Input/waterCover_100m/'

# hotMonthes = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\hotMonth\hotMonth_city.csv')

# hotMonthes['hotmonth']=np.round(hotMonthes[hotMonthes.columns[1:]].mean(axis=1))

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
        grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') #+ tf.imread(
            #shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
        grassC = smooth_2d_array(grassC, window_size=10)
        # shrubC = tf.imread(shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
        # shrubC = smooth_2d_array(shrubC, window_size=5)
        cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
        cropC = smooth_2d_array(cropC, window_size=10)
        # bareC = tf.imread(bareC_folder + 'bareC_' + str(i) + '.0.tif')
        # bareC = smooth_2d_array(bareC, window_size=5)
        # buildC = tf.imread(impervious_folder + 'imperviousC_' + str(i) + '.0.tif')
        buildC = tf.imread(buildC_folder + 'buildC_' + str(i) + '.0.tif')
        buildC = smooth_2d_array(buildC, window_size=10)
        waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
        waterC = smooth_2d_array(waterC, window_size=10)
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
        if valid_ratio < 0.2: continue
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
        X.columns = ['tree', 'grass', 'crop', 'impervious', 'water', 'dem', 'tair']

        if X.shape[0]<100: continue

        model = xgb.XGBRegressor(max_depth=2, min_child_weight=11, subsample=0.65, colsample_bytree=0.65, eta=0.15,
                                 tree_method='hist', max_delta_step=6, eval_metric='auc')  # lower max_depth, larger
        # min_child and gamma, model more conservative
        model.fit(X.iloc[:, 0:-1], X.iloc[:, -1])
        predic = model.predict(X.iloc[:, 0:-1]).tolist()
        r = st.linregress(X.iloc[:, -1], predic).rvalue
        ###
        # plt.figure(); plt.hist(X.iloc[:,0], bins = 20)

        X1 = X.iloc[0:101, :] * 1
        X1.iloc[:, 0] = np.linspace(1, 0, 101)
        X1.iloc[:, [1, 2, 4]] = 0
        X1.iloc[:, 3] = np.linspace(0, 1, 101)
        X1.iloc[:, 5] = np.nanmean(dem)
        predict1 = model.predict(X1.iloc[:, 0:-1]).tolist()
        # plt.figure(), plt.plot(predict1)
        ts_tree = np.nanmean(predict1[0:2]) + np.nanmean(tair_bg)

        X1.iloc[:, 1] = np.linspace(1, 0, 101)
        X1.iloc[:, [0, 2, 4]] = 0
        X1.iloc[:, 3] = np.linspace(0, 1, 101)
        X1.iloc[:, 5] = np.nanmean(dem)
        predict1 = model.predict(X1.iloc[:, 0:-1]).tolist()
        # plt.figure(); plt.plot(predict1)
        ts_grass = np.nanmean(predict1[0:2]) + np.nanmean(tair_bg)

        X1.iloc[:, 2] = np.linspace(1, 0, 101)
        X1.iloc[:, [0, 1, 4]] = 0
        X1.iloc[:, 3] = np.linspace(0, 1, 101)
        X1.iloc[:, 5] = np.nanmean(dem)
        predict1 = model.predict(X1.iloc[:, 0:-1]).tolist()
        # plt.figure(); plt.plot(predict1)
        ts_crop = np.nanmean(predict1[0:2]) + np.nanmean(tair_bg)
        ts_build = predict1[-1]+np.nanmean(tair_bg)

        ce_tree = ts_tree - ts_build  # negative is cooling
        ce_grass = ts_grass - ts_build  # negative is cooling
        ce_crop = ts_crop - ts_build  # negative is cooling

        # ce = mean_slope(data,values)
        ce = [ce_tree, ts_tree, ce_grass, ts_grass, ce_crop, ts_crop, ts_build, r]
        # ce = [ce_tree, ce_grass, ce_crop, r]
        CE.append([i] + [month] + ce)
    # CE.append([i] + ce)
    print(i)

CE = np.array(CE)
CE = pd.DataFrame(CE)
CE.columns = ['ID', 'month','CEts_tree','ts_tree','CEts_grass', 'ts_grass','CEts_cropland','ts_cropland','ts_build', 'r']
CE.to_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3.csv', index=False)
