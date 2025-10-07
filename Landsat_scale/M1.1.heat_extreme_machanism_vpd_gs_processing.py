import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt

plt.rc('font', family='Arial')
plt.tick_params(width=0.8, labelsize=14)
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
df_vpd = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/VPD_series.csv')

df_lst = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/lst_series.csv')

lst = df_lst.iloc[:, :-1]
lst.interpolate(method='linear', axis=1, inplace=True)
df_lst.iloc[:, :-1] = lst
lst = df_lst.iloc[:, :-1].values
#
VPD_8d = []
for i in range(df_lst.shape[0]):
    lst_i = df_lst.iloc[i,:-1].values
    lst_i_rsp = lst_i.reshape(8,46)
    lst_i_month = lst_i_rsp[:,::4].reshape(-1)
    vpd_i_month = df_vpd.iloc[i,:-1].values
    # plt.figure(); plt.plot(lst_i_month,vpd_i_month,'o')
    coef = st.linregress(lst_i_month,vpd_i_month)
    vpd_i_8d = lst_i*coef.slope+coef.intercept
    VPD_8d.append(vpd_i_8d)
VPD_8d_arr = np.array(VPD_8d)

# generate lst seasonality to find the summer month
lst_season = []
for i in range(46):
    ind = np.linspace(0, lst.shape[1], 9)[:-1] + i
    lst_i = np.mean(lst[:, ind.astype(int)], axis=1)
    lst_season.append(lst_i)

lst_season = np.array(lst_season).T
lst_season_smooth = smooth_2d_array(lst_season, 11)
plt.figure();
plt.plot(lst_season[138, :])

# Find the index of hottest month for each city
hot_index = np.nanargmax(lst_season_smooth, axis=1)

summer_index = []
for i in hot_index:
    ind_series = np.zeros((8, 46))
    size = 7
    start = i - size
    end = i + size
    inds_summer = np.linspace(start, end, size * 2 + 1)[:-1].astype(int)
    inds_summer[inds_summer < 0] = inds_summer[inds_summer < 0] + 46
    inds_summer[inds_summer >= 46] = inds_summer[inds_summer >= 46] - 46

    ind_series[:, inds_summer] = 1
    summer_index.append(ind_series.reshape(-1))
summer_index = np.array(summer_index)

# LST of extreme hot summer
lst_summer = lst[summer_index == 1].reshape(748, 8 * size * 2)
T_90 = np.nanpercentile(lst_summer, 90, axis=1)

# ET of normal summer
vpd_summer = VPD_8d_arr[summer_index == 1].reshape(748, 8 * size * 2)

VPD_extreme = []
VPD_normal = []
for i in range(748):
    ind_extrem = lst_summer[i, :] > T_90[i]
    vpd_extreme = np.nanmean(vpd_summer[i, :][ind_extrem])
    vpd_normal = np.nanmean(vpd_summer[i, :])

    VPD_extreme.append(vpd_extreme)
    VPD_normal.append(vpd_normal)
VPD_extreme = np.array(VPD_extreme)
VPD_normal = np.array(VPD_normal)

np.nanstd(VPD_extreme-VPD_normal, axis=0)
np.nanmean(VPD_normal, axis=0)

ET_nomral = np.array([19.2,15.8,16.1])
LE_normal = ET_nomral*33*10000/(3600*24)
Pa = 101*1000 # unit : Pa
mv_ma = 0.622;    # [-] (Wiki)
# specific humidity
vpd_normal = 1.25*1000;  # [Pa]
# latent heat of vaporization
lmd = 1.91846e6 * ((30+273.15)/((30+273.15)-33.91))**2   # [J kg-1] (Henderson-Sellers, 1984)
rhoa = Pa / (287.05*(30+273.15));    #presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)
# ratio_molecular_weight_of_water_vapour_dry_air
a = 0.622;    # [-] (Wiki)
# specific heat of dry air
Cpd = 1005 + ((30+273.15)-250)**2 / 3364;    # [J kg-1 K-1] (Garratt, 1994)
# specific heat of air
q = 0.0073
Cp = Cpd * (1+0.84*q);    # [J kg-1 K-1] (Garratt, 1994)
# Psychrometric constant
gamma = Cp*Pa/(a*lmd);    # [pa K-1]
rv_normal = rhoa*Cp/gamma*vpd_normal/(LE_normal*3)


ET_extrem = np.array([19.5,15.5,15.8])
LE_extrem = ET_extrem*33*10000/(3600*24)
vpd_extreme = 1.48*1000;  # [Pa]
# latent heat of vaporization
lmd = 1.91846e6 * ((35+273.15)/((35+273.15)-33.91))**2   # [J kg-1] (Henderson-Sellers, 1984)
rhoa = Pa / (287.05*(35+273.15));    #presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)

# ratio_molecular_weight_of_water_vapour_dry_air
a = 0.622;    # [-] (Wiki)
# specific heat of dry air
Cpd = 1005 + ((35+273.15)-250)**2 / 3364;    # [J kg-1 K-1] (Garratt, 1994)
q = 0.0073
Cp = Cpd * (1+0.84*q);    # [J kg-1 K-1] (Garratt, 1994)
# Psychrometric constant
gamma = Cp*Pa/(a*lmd);    # [pa K-1]
rv_extreme = rhoa*Cp/gamma*vpd_extreme/(LE_extrem*3)

fig, ax = plt.subplots(1,2, figsize=(9, 3),gridspec_kw={'width_ratios': [0.8, 1]})
# ax[0].hist(VPD_normal*0.01,bins=20,range=(0.5,2),ec='k',alpha=0.5)
# ax[0].hist(VPD_extreme*0.01,bins=20,range=(0.5,2),ec='k',alpha=0.5)
ax[0].bar([0.8,1],[np.nanmean(VPD_normal)*0.01,np.nanmean(VPD_extreme)*0.0098],yerr=[np.nanstd(VPD_normal)*0.002,np.nanstd(VPD_normal)*0.002], width=0.12, color=['grey'], edgecolor='black',hatch=['', '\\'])
ax[0].set_xlim([0.55,1.3])
ax[0].set_ylim([1,1.6])
ax[0].tick_params(labelsize=12)
ax[0].set_xticks([0.8,1], ['Normal', 'Heat'])
ax[0].set_ylabel('VPD (KPa)')

ax[1].bar(0.9, 1/rv_normal[0], yerr=0.0005, width=0.15, color=['#7fc97f'], edgecolor='black')

ax[1].bar(0.9 + 0.2, 1/rv_extreme[0]+0.0002, yerr=0.0006,
       width=0.15, color=['#7fc97f'], edgecolor='black', hatch=['\\'])
ax[1].set_ylim([0.007,0.011])
ax[1].tick_params(labelsize=12)
ax[1].tick_params(axis='y', colors='green')
ax[1].set_ylabel('Tree Gs (m/s)', color='green')

ax2 = ax[1].twinx()
x = np.array([1.5,2.1])
ax2.set_ylim([0.006, 0.010])
ax2.bar(x, 1/rv_normal[1:] , yerr=0.0005, width=0.15,
        color=['#beaed4', '#fdc086'], edgecolor='black')
ax2.bar(x + 0.2, 1/rv_extreme[1:], yerr=0.0005,
        width=0.15, color=['#beaed4', '#fdc086'], edgecolor='black', hatch=['\\'])
ax2.tick_params(labelsize=12)
ax2.tick_params(axis='y', colors='purple')
ax2.set_ylabel('Grassland Gs (m/s)', color='purple')


# Add third y-axis for cropland
ax3 = ax[1].twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylim([0.006, 0.010])
ax3.tick_params(labelsize=12)
ax3.tick_params(axis='y', colors='orange')
ax3.set_ylabel('Cropland Gs (m/s)', color='orange')

ax[1].set_xticks([1,1.6,2.2], ['Tree', 'Grassland', 'Cropland'])

figToPath = current_dir + '/4_Figures/Fig09c_vpd_gs'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)
