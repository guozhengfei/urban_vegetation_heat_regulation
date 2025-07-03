import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
import geopandas as gpd
from scipy.ndimage import convolve1d
import scipy.io
import seaborn as sns
from scipy.optimize import curve_fit

cities_ID = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')['ID'].astype(int)

treeC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\tree_cover\\'

grassC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\grass_cover\\'

cropC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\crop_cover\\'

shrubC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\shrub_cover\\'

buildC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\building_cover\\'

bareC_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\bare_cover\\'

waterC_folder = r'D:\Projects\Postdoc urban greening\Data\waterFraction1000m\\'

urbanRaster_folder = r'D:\Projects\Postdoc urban greening\Data\urban_tiff\Urbanized_and_urbanization_area\\'

landtype_frac = []
for i in cities_ID[1:]:
    treeC = tf.imread(treeC_folder + 'treeC_' + str(i) + '.0.tif')
    shrubC = tf.imread(shrubC_folder + 'shrubC_' + str(i) + '.0.tif')
    grassC = tf.imread(grassC_folder + 'grassC_' + str(i) + '.0.tif')
    cropC = tf.imread(cropC_folder + 'cropC_' + str(i) + '.0.tif')
    bareC = tf.imread(bareC_folder + 'bareC_' + str(i) + '.0.tif')
    buildC = tf.imread(buildC_folder + 'buildingC_' + str(i) + '.0.tif')
    waterC = tf.imread(waterC_folder + 'waterC_' + str(i) + '.0.tif')
    urban = tf.imread(urbanRaster_folder + 'fvc_'+str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan

    waterC = np.nanmean(waterC * urban)
    treeC = np.nanmean(treeC * urban)
    grassC = np.nanmean(grassC * urban)
    shrubC = np.nanmean(shrubC * urban)
    cropC = np.nanmean(cropC * urban)
    buildC = np.nanmean(buildC * urban)
    bareC = np.nanmean(bareC * urban)

    urbanSize = np.nansum(urban)
    frac = [treeC, shrubC, grassC, cropC, bareC, buildC, waterC, urbanSize]
    landtype_frac.append([i] + frac)
    print(i)

landtype_frac = pd.DataFrame(np.array(landtype_frac))
landtype_frac.columns = ['ID','tree','shrub','grass','crop','bare','build','water','urbanS']
landtype_frac.to_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\landtype_fraction.csv')
treeC_all = np.sum(landtype_frac['tree']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

shrubC_all = np.sum(landtype_frac['shrub']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

grassC_all = np.sum(landtype_frac['grass']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

cropC_all = np.sum(landtype_frac['crop']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

bareC_all = np.sum(landtype_frac['bare']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

buildC_all = np.sum(landtype_frac['build']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

waterC_all = np.sum(landtype_frac['water']*landtype_frac['urbanS'])/landtype_frac['urbanS'].sum()

import seaborn as sns
fig, ax = plt.subplots(1,figsize=(5.2,4))
sns.violinplot(data=landtype_frac.iloc[:,1:-1],scale='count',order=['build','tree','grass','crop','bare','water','shrub'])

figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Figs1_vegetation_fraction'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

# treeC_all + grassC_all+ shrubC_all+ cropC_all+ bareC_all + buildC_all + waterC_all