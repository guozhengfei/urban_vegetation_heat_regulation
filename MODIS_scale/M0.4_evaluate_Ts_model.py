import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import geopandas as gpd
from scipy.ndimage import convolve1d
import scipy.io
import seaborn as sns
from scipy.optimize import curve_fit

from scipy.stats import variation
from scipy.stats import theilslopes
import pymannkendall as mk

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

def remove_holes(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result>0]=1
    return result

def remove_edge(array):
    array1 = array * 1
    array1[1:, :] = array[:-1, :]  # Shift elements up
    array2 = array * 1
    array2[:-1, :] = array[1:, :]  # Shift elements down
    array3 = array * 1
    array3[:, 1:] = array[:, :-1]  # Shift elements left
    array4 = array * 1
    array4[:, :-1] = array[:, 1:]  # Shift elements right
    array5 = array * 1
    array5[:-1, 1:] = array[1:, :-1]  # Shift elements up-left
    array6 = array * 1
    array6[:-1, :-1] = array[1:, 1:]  # Shift elements up-right
    array7 = array * 1
    array7[1:, 1:] = array[:-1, :-1]  # Shift elements down-right
    array8 = array * 1
    array8[1:, :-1] = array[:-1, 1:]  # Shift elements down-left
    stacked_array = np.stack([array1, array2, array3, array4, array5, array6, array7, array8], axis=0)
    result = np.nanmean(stacked_array, axis=0)
    result[result<1]=0
    return result

def minus_mean_3d(array):
    for i in range(array.shape[2]):
        mean_i = np.nanmean(array[:,:,i])
        array[:,:,i] = array[:,:,i]-mean_i
    return array

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
    mask = np.isnan(urban) | np.isnan(Z)
    X1 = X[~mask]
    Y1 = Y[~mask]
    Z1 = Z[~mask]
    p0 = [np.mean(Z1), 1, 1]  # Initial guess for the parameters
    fit_params, _ = curve_fit(gaussian_func, (X1, Y1), Z1, p0)
    Z1_fit = gaussian_func((X1, Y1), *fit_params)
    st.linregress(Z1_fit,Z1)
    Z_fit = gaussian_func((X, Y), *fit_params)
    return Z_fit

## main ##
cities_ID = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\tree_cover\\'

grassC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\grass_cover\\'

cropC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\crop_cover\\'

shrubC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\shrub_cover\\'

buildC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\building_cover\\'

bareC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\bare_cover\\'

TairMaxTiff_folder = r'D:\Projects\Postdoc urban greening\Data\TsMax\\'

urbanRaster_folder = r'D:\Projects\Postdoc urban greening\Data\urban_tiff\Urbanized_and_urbanization_area\\'

impervious_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\Impervious\\'

dem_folder = r'D:\Projects\Postdoc urban greening\Data\DEM\\'

pop_folder = r'D:\Projects\Postdoc urban greening\Data\Population_density\\'

water_folder = r'D:\Projects\Postdoc urban greening\Data\waterFraction1000m\\'

hotMonthes = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\hotMonth\hotMonth_city.csv')

hotMonthes['hotmonth']=np.round(hotMonthes[hotMonthes.columns[1:]].mean(axis=1))

R = []
for i in cities_ID:
    try:
        tairMax = tf.imread(TairMaxTiff_folder + 'Ts_' + str(i) + '.0.tif')
    except FileNotFoundError: continue
    if np.isnan(np.nanmean(tairMax)): continue

    hotmonth = hotMonthes[hotMonthes['ID'] == i]['hotmonth'].values[0] - 1
    month0 = np.int16(hotmonth)
    month0 = np.array([month0-1, month0, month0+1])
    # tairHot = np.nanmean(tairMax[:, :, 12 * 17 + month0],axis=2)
    tairHot = np.nanmean(tairMax[:, :, 12 * 17:], axis=2) # 2 years average

    # plt.figure(); plt.imshow(tairHot)
    treeC = tf.imread(treeC_folder + 'treeC_' + str(i) + '.0.tif')
    treeC = smooth_2d_array(treeC, window_size=5)
    grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif') + tf.imread(shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
    grassC = smooth_2d_array(grassC, window_size=5)
    # shrubC = tf.imread(shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
    # shrubC = smooth_2d_array(shrubC, window_size=5)
    cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
    cropC = smooth_2d_array(cropC, window_size=5)
    # bareC = tf.imread(bareC_folder + 'bareC_' + str(i) + '.0.tif')
    # bareC = smooth_2d_array(bareC, window_size=5)
    buildC = tf.imread(impervious_folder + 'imperviousC_' + str(i) + '.0.tif')
    buildC = smooth_2d_array(buildC, window_size=5)
    waterC = tf.imread(water_folder + 'waterC_' + str(i) + '.0.tif')
    waterC = smooth_2d_array(waterC, window_size=5)

    Dem = tf.imread(dem_folder + 'dem_' + str(i) + '.0.tif')
    pop = tf.imread(pop_folder + 'pop_' + str(i) + '.0.tif')
    urban = tf.imread(urbanRaster_folder + 'fvc_'+str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    # # # remove the hole within the urban area
    # urban = remove_holes(remove_holes(remove_holes(remove_holes(remove_holes(urban)))))
    # urban = remove_edge(remove_edge(remove_edge(remove_edge(remove_edge(urban)))))
    urban[urban==0] = np.nan

    tair_bg = background_climate(tairHot,Dem,urban)

    Dem = Dem*urban
    Dem[Dem > (np.nanmean(Dem)+3*np.nanstd(Dem))] = np.nan

    waterC = waterC * urban
    treeC = treeC * urban
    grassC = grassC * urban
    # shrubC = shrubC * urban
    cropC = cropC * urban
    buildC = buildC * urban
    # bareC = bareC * urban

    tree = treeC.reshape(-1)
    # shrub = shrubC.reshape(-1)
    grass = grassC.reshape(-1)
    crop = cropC.reshape(-1)
    build = buildC.reshape(-1)
    # bare = bareC.reshape(-1)
    water = waterC.reshape(-1)
    # tree[tree < np.nanpercentile(tree, 1)] = np.nan
    # tair = tairHot.reshape(-1)
    tair = (tairHot-tair_bg).reshape(-1)
    dem = Dem.reshape(-1)
    mask = np.isnan(tree) | np.isnan(grass) | np.isnan(water) | np.isnan(tair) | np.isnan(dem)

    # plt.figure(); plt.hist(water[~mask], bins = 50)
    # Prepare the data
    X = np.array([tree[~mask], grass[~mask],  crop[~mask], build[~mask],  water[~mask], dem[~mask], tair[~mask]]).T # shrub[~mask], build[~mask],,
    X = pd.DataFrame(X)
    X.columns = ['tree', 'grass',  'crop', 'building',  'water', 'dem', 'tair'] #'shrub', 'building',
    folds = 10
    part_num = int(X.shape[0] / folds)
    # devide into 10 parts
    X_shuffled = X.sample(frac=1, random_state=42)
    pred = []; real = []
    for n in range(folds):
        indexs = np.int16(part_num*n+np.linspace(0,part_num-1,part_num))
        extracted_rows = X_shuffled.loc[indexs.tolist(),:] # used for test
        remaining_rows = X_shuffled.drop(extracted_rows.index) # used for train
        train_features = remaining_rows.iloc[:,0:X.shape[1]-1]
        train_labels = remaining_rows.iloc[:,X.shape[1]-1]
        test_features = extracted_rows.iloc[:, 0:X.shape[1]-1]
        test_lables = extracted_rows.iloc[:, X.shape[1]-1]
        model = xgb.XGBRegressor(max_depth=2, min_child_weight=11, subsample=0.65, colsample_bytree=0.65, eta=0.15,
                                 tree_method='hist', max_delta_step=6, eval_metric='auc')  # lower max_depth, larger
        # min_child and gamma, model more conservative
        model.fit(train_features, train_labels)
        predic_labels = model.predict(test_features).tolist()
        real = real + test_lables.values.tolist()
        pred= pred + predic_labels
    # plt.figure(); plt.plot(real,pred,'.')
    r = st.linregress(real,pred).rvalue
    R.append([i,r])
    print(i)

R = np.array(R)

np.nansum(R[:,1]>0.5)/747
plt.figure(); plt.hist(IQR_filter(R[:, 1]), 50)

scipy.io.savemat(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\model_performance_2020_yearly_Ts.mat', {'R': R})

correlation_matrix = X.iloc[:,0:-1].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)