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

# shrubC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\shrub_cover\\'

buildC_folder = current_dir + '/1_Input/buildCover_100m/'

# bareC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\bare_cover\\'

TairMaxTiff_folder_extrem = current_dir + '/1_Input/LST_extreme/85th/'

TairMaxTiff_folder_normal = current_dir + '/1_Input/LST/'

urbanRaster_folder = current_dir + '/1_Input/urbanRaster_100m/'

# impervious_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\Impervious\\'

dem_folder = current_dir + '/1_Input/DEM_100m/'

# pop_folder = r'D:\Projects\Postdoc urban greening\Data\Population_density\\'

water_folder = current_dir + '/1_Input/waterCover_100m/'

# hotMonthes = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\hotMonth\hotMonth_city.csv')

# hotMonthes['hotmonth']=np.round(hotMonthes[hotMonthes.columns[1:]].mean(axis=1))

CE = []
for i in cities_ID[1:]:
    if i in [703,689]: continue
    urban = tf.imread(urbanRaster_folder + 'urban_' + str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan

    try:
        tairMax_extreme = tf.imread(TairMaxTiff_folder_extrem + 'Landsat_' + str(i) + '.0.tif')
    except FileNotFoundError:
        continue


    tairHot = tairMax_extreme
    valid_ratio = np.sum(~np.isnan(tairHot*urban)) / np.sum(~np.isnan(urban))
    # plt.figure(); plt.imshow(urban_3d[:,:,1])
    if valid_ratio < 0.6: continue
    # gap-fill

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
    tair_bg = 0

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

    # bareC = bareC * urban
    # pop = pop * urban

    # pop = pop.reshape(-1)
    tree = treeC.reshape(-1)
    grass = grassC.reshape(-1)
    crop = cropC.reshape(-1)
    build = buildC.reshape(-1)
    water = waterC.reshape(-1)
    tair = (tairHot - tair_bg).reshape(-1)

    tair[tair>np.nanmean(tair)+3*np.nanstd(tair)]=np.nan
    tair[tair<np.nanmean(tair)-3*np.nanstd(tair)]=np.nan

    dem = Dem.reshape(-1)
    mask = np.isnan(tree) | np.isnan(grass) | np.isnan(water) | np.isnan(tair) | np.isnan(dem)

    # plt.figure(); plt.hist(tree[~mask], bins = 50)
    # Prepare the data
    X = np.array([tree[~mask], grass[~mask], crop[~mask], build[~mask], water[~mask], dem[~mask], tair[~mask]]).T  # ,
    X = pd.DataFrame(X)
    X.columns = ['tree', 'grass', 'crop', 'impervious', 'water', 'dem', 'tair']

    # plt.figure(); plt.plot(X['tree'],X['tair'],'.')

    if X.shape[0]<100: continue

    model = xgb.XGBRegressor(max_depth=2, min_child_weight=11, subsample=0.65, colsample_bytree=0.65, eta=0.15,
                             tree_method='hist', max_delta_step=6, eval_metric='auc')  # lower max_depth, larger
    # min_child and gamma, model more conservative
    model.fit(X.iloc[:, 0:-1], X.iloc[:, -1])
    predic = model.predict(X.iloc[:, 0:-1]).tolist()
    r = st.linregress(X.iloc[:, -1], predic).rvalue
    ###
    # plt.figure(); plt.plot(X.iloc[:, -1], predic,'.')

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
    CE.append([i] + ce)
    print(i)

CE = np.array(CE)
CE = pd.DataFrame(CE)
CE.columns = ['ID', 'CEts_tree', 'ts_tree','CEts_grass', 'ts_grass', 'CEts_cropland', 'ts_cropland', 'ts_build', 'r']
CE.to_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_summer_extreme_85th.csv', index=False)
plt.figure(); plt.hist(IQR_filter(CE.iloc[:, -1]), 20, ec='black')

np.sum(CE.iloc[:, 5] > 0) / np.sum(~np.isnan(CE.iloc[:, 5]))
np.nanmean(CE, axis=0)