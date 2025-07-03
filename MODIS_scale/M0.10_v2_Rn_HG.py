import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
import geopandas as gpd
from scipy.ndimage import convolve1d
import scipy.io
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import curve_fit

# plt.figure(); plt.plot(data[:, 3],values[:, 3] + shap_values.base_values,'.')

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

Ts_folder = r'D:\Projects\Postdoc urban greening\Data\TsMax\\'

sw_folder = r'D:\Projects\Postdoc urban greening\Data\shortwave\\'

lw_folder = r'D:\Projects\Postdoc urban greening\Data\longwave\\'

alb_folder = r'D:\Projects\Postdoc urban greening\Data\Albedo\\'

LE_folder = r'D:\Projects\Postdoc urban greening\Data\ET_canopy\\'

urbanRaster_folder = r'D:\Projects\Postdoc urban greening\Data\urban_tiff\Urbanized_and_urbanization_area\\'

impervious_folder = r'D:\Projects\Postdoc urban greening\Data\FractionFrom10m\Impervious\\'

dem_folder = r'D:\Projects\Postdoc urban greening\Data\DEM\\'

pop_folder = r'D:\Projects\Postdoc urban greening\Data\Population_density\\'

water_folder = r'D:\Projects\Postdoc urban greening\Data\waterFraction1000m\\'

hotMonthes = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\hotMonth\hotMonth_city.csv')
hotMonthes['hotmonth']=np.round(hotMonthes[hotMonthes.columns[1:]].mean(axis=1))

energy_ratios = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\energy_ratio_csv.csv')
Rad = []
for i in cities_ID[1:]:

    # Alb = np.nanmean(tf.imread(alb_folder + 'albedo_' + str(i) + '.0.tif')[:, :, -13:-1], axis=2)
    #
    # # Rn = (1-alb)*SW_d+emis*LW_d - sig*emis*Ts**4
    # Rn = (1 - 0.001 * Alb) * SW + 0.97 * LW - 0.97 * 5.67037442 * 10 ** -8 * (Ts + 273.15) ** 4

    # LE0 = np.nanmean(tf.imread(LE_folder + 'ET_' + str(i) + '.0.tif')[:, :, -13:-1], axis=2)/(3600*24)
    #
    # LE = LE0*10**6*(2.501-2.361*0.001*(Ts))
    # # plt.figure(); plt.imshow(Ts)
    # HG = Rn - LE

    SW = np.nanmean(tf.imread(sw_folder + 'sw_' + str(i) + '.0.tif')[:, :, -13:-1], axis=2) / (
                3600 * 24)  # *ratios['SW_ratio'].values

    LW = np.nanmean(tf.imread(lw_folder + 'lw_' + str(i) + '.0.tif')[:, :, -13:-1], axis=2) / (
                3600 * 24)  # *ratios['LW_ratio'].values

    urban = tf.imread(urbanRaster_folder + 'fvc_' + str(i) + '.0.tif').astype(float)
    urban[urban != i] = 0
    urban[urban == i] = 1
    urban[urban == 0] = np.nan

    SW_mean_dw = np.nanmean(SW*urban)
    LW_mean_dw = np.nanmean(LW*urban)
    rad = [i,SW_mean_dw, LW_mean_dw]
    Rad.append(rad)
    print(i)

Rad = np.array(Rad)
Rad = pd.DataFrame(Rad)
Rad.columns = ['ID', 'SW_dw', 'LW_dw']
Rad.to_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Rad_2020_yearly_v5.csv', index=False)
