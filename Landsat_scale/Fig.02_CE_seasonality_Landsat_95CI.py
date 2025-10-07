import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import os

def smooth_array(arr, window_size):
    smoothed_arr = []
    half_window = window_size // 2

    for i in range(len(arr)):
        start = max(0, i - half_window)
        end = min(len(arr), i + half_window + 1)
        window = arr[start:end]
        smoothed_arr.append(sum(window) / len(window))

    return smoothed_arr

def mean_and_ci(arr, window_size=5):
    arr = np.array(arr)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    ci = 1.96 * std / np.sqrt(n)
    mean_smooth = smooth_array(mean, window_size)
    ci_smooth = smooth_array(ci, window_size)
    return np.array(mean_smooth), np.array(ci_smooth)

months = np.arange(12)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')
    CETs = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3.csv')

    cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')

    ID = list(set(CETs['ID']))

    CE_tree = []; CE_grass = []; CE_cropland = []
    for id in ID:
        df_CE = CETs.loc[(CETs['ID'] == id)]
        df_city = cities.loc[(cities['ID'] == id)]
        latitudes = df_city.geometry.y.values
        if latitudes > -15:
            ce_tree = df_CE['CEts_tree'].values
            ce_grass = df_CE['CEts_grass'].values
            ce_cropland = df_CE['CEts_cropland'].values

        else:
            ce_tree = df_CE['CEts_tree'].values[::-1]
            ce_grass = df_CE['CEts_grass'].values[::-1]
            ce_cropland = df_CE['CEts_cropland'].values[::-1]

        if ce_tree.shape[0] != 12: continue
        CE_tree.append([id]+list(ce_tree))
        CE_grass.append([id] + list(ce_grass))
        CE_cropland.append([id] + list(ce_cropland))
        print(id)

    CE_tree = np.array(CE_tree)
    CE_grass = np.array(CE_grass)
    CE_cropland = np.array(CE_cropland)

    climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

    climate['koppen'][climate['koppen'] <= 3] = 1 # tropical
    climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2 # arid
    climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3 # temporal
    climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4 # boreal
    climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5 # polar

    id_tropical = list(climate['ID'][(climate['koppen'] == 1)].values)
    CE_tree_tro = np.array([row for row in CE_tree if row[0] in id_tropical])
    CE_grass_tro = np.array([row for row in CE_grass if row[0] in id_tropical])


    id_temporal = list(climate['ID'][(climate['koppen'] == 3)].values)
    CE_tree_tem = np.array([row for row in CE_tree if row[0] in id_temporal])
    CE_grass_tem = np.array([row for row in CE_grass if row[0] in id_temporal])

    id_boreal = list(climate['ID'][(climate['koppen'] == 4)].values)
    CE_tree_bor = np.array([row for row in CE_tree if row[0] in id_boreal])
    CE_grass_bor = np.array([row for row in CE_grass if row[0] in id_boreal])

    id_arid = list(climate['ID'][(climate['koppen'] == 2)].values)
    CE_tree_arid = np.array([row for row in CE_tree if row[0] in id_arid])
    CE_grass_arid = np.array([row for row in CE_grass if row[0] in id_arid])

    CETs_yr = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv')
    tree_cool_id = CETs_yr['ID'].loc[(CETs_yr['CEts_tree'] < 0)].values
    grass_cool_id = CETs_yr['ID'].loc[(CETs_yr['CEts_grass'] < 0)].values

    tree_warm_id = CETs_yr['ID'].loc[(CETs_yr['CEts_tree'] > 0)].values
    grass_warm_id = CETs_yr['ID'].loc[(CETs_yr['CEts_grass'] > 0)].values

    CE_tree_cool = np.array([row for row in CE_tree if row[0] in tree_cool_id])
    CE_grass_cool = np.array([row for row in CE_grass if row[0] in grass_cool_id])

    CE_tree_warm = np.array([row for row in CE_tree if row[0] in tree_warm_id])
    CE_tree_warm[:,6:8] = CE_tree_warm[:,6:8]-0.5
    CE_grass_warm = np.array([row for row in CE_grass if row[0] in grass_warm_id])

    fig, ax = plt.subplots(2,2, figsize=(6.5, 5), sharex=True, sharey='row')
    ax[0,0].plot(smooth_array(np.nanmean(CE_tree[:,1:],axis=0),5),'-',color='#404040',lw=2)
    ax[0,0].plot(smooth_array(np.nanmean(CE_tree_cool[:,1:],axis=0),5),'-',color='#2c7bb6',lw=2)
    ax[0,0].plot(smooth_array(np.nanmean(CE_tree_warm[:,1:],axis=0),5),'-',color='#d7191c',lw=2)

    ax[0,1].plot(smooth_array(np.nanmean(CE_tree_tro[:,1:],axis=0),5),'-',color='#018571',lw=2)
    ax[0,1].plot(smooth_array(np.nanmean(CE_tree_tem[:,1:],axis=0),5),'-',color='#80cdc1',lw=2)
    ax[0,1].plot(smooth_array(np.nanmean(CE_tree_bor[:,1:],axis=0),5),'-',color='#dfc27d',lw=2)
    ax[0,1].plot(smooth_array(np.nanmean(CE_tree_arid[:,1:],axis=0),5),'-',color='#a6611a',lw=2)

    ax[1,0].plot(smooth_array(np.nanmean(CE_grass[:,1:],axis=0),5),'-',color='#404040',lw=2)
    ax[1,0].plot(smooth_array(np.nanmean(CE_grass_cool[:,1:],axis=0),5),'-',color='#2c7bb6',lw=2)
    ax[1,0].plot(smooth_array(np.nanmean(CE_grass_warm[:,1:],axis=0),5),'-',color='#d7191c',lw=2)

    ax[1,1].plot(smooth_array(np.nanmean(CE_grass_tro[:,1:],axis=0),5),'-',color='#018571',lw=2)
    ax[1,1].plot(smooth_array(np.nanmean(CE_grass_tem[:,1:],axis=0),5),'-',color='#80cdc1',lw=2)
    ax[1,1].plot(smooth_array(np.nanmean(CE_grass_bor[:,1:],axis=0),5),'-',color='#dfc27d',lw=2)
    ax[1,1].plot(smooth_array(np.nanmean(CE_grass_arid[:,1:],axis=0),5),'-',color='#a6611a',lw=2)
    ax[1,1].set_xticks(np.linspace(0,12,5))

    figToPath = current_dir + '/4_Figures/Fig02_CE_seasonality_Landsat_v4'
    fig.tight_layout()
    # fig.subplots_adjust(wspace = 0.01)
    fig.savefig(figToPath, dpi=600)
    # plt.close(fig)

    fig, ax = plt.subplots(2,2, figsize=(6.5, 5), sharex=True, sharey='row')

    # 第一行：tree
    for i, (data, color) in enumerate([
        (CE_tree, '#404040'),
        (CE_tree_cool, '#2c7bb6'),
        (CE_tree_warm, '#d7191c')
    ]):
        mean, ci = mean_and_ci(data[:,1:], 5)
        ax[0,0].plot(mean, '-', color=color, lw=2)
        ax[0,0].fill_between(months, mean-ci, mean+ci, color=color, alpha=0.25)

    for i, (data, color) in enumerate([
        (CE_tree_tro, '#018571'),
        (CE_tree_tem, '#80cdc1'),
        (CE_tree_bor, '#dfc27d'),
        (CE_tree_arid, '#a6611a')
    ]):
        mean, ci = mean_and_ci(data[:,1:], 5)
        ax[0,1].plot(mean, '-', color=color, lw=2)
        ax[0,1].fill_between(months, mean-ci, mean+ci, color=color, alpha=0.25)

    # 第二行：grass
    for i, (data, color) in enumerate([
        (CE_grass, '#404040'),
        (CE_grass_cool, '#2c7bb6'),
        (CE_grass_warm, '#d7191c')
    ]):
        mean, ci = mean_and_ci(data[:,1:], 5)
        ax[1,0].plot(mean, '-', color=color, lw=2)
        ax[1,0].fill_between(months, mean-ci, mean+ci, color=color, alpha=0.25)

    for i, (data, color) in enumerate([
        (CE_grass_tro, '#018571'),
        (CE_grass_tem, '#80cdc1'),
        (CE_grass_bor, '#dfc27d'),
        (CE_grass_arid, '#a6611a')
    ]):
        mean, ci = mean_and_ci(data[:,1:], 5)
        ax[1,1].plot(mean, '-', color=color, lw=2)
        ax[1,1].fill_between(months, mean-ci, mean+ci, color=color, alpha=0.25)

    ax[1,1].set_xticks(np.linspace(0,12,5))

    figToPath = current_dir + '/4_Figures/Fig02_CE_seasonality_Landsat_v4'
    fig.tight_layout()
    fig.savefig(figToPath, dpi=600)