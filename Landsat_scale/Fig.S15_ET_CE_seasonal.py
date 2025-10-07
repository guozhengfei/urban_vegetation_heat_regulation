import matplotlib
matplotlib.use('Qt5Agg')
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import os
import numpy as np
import scipy.stats as st
from sklearn.linear_model import LinearRegression

if __name__=='__main__':
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
    relative_path1 = '/2_Output/ET_csv_urban_751.csv' # TRC based on air temperature
    ET = pd.read_csv(current_dir+relative_path1) # v5,v4.2
    climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

    climate['koppen'][climate['koppen'] <= 3] = 1
    climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
    climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
    climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
    climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

    relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv'
    CETs = pd.read_csv(current_dir + relative_path)  # v5,v4.2
    cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')
    cities = pd.merge(cities, CETs, on='ID')
    cities = pd.merge(cities, climate, on='ID')
    cities_warm = cities.loc[cities['CEts_grass'] > 0]


    def smooth_array(arr, window_size):
        smoothed_arr = []
        half_window = window_size // 2

        for i in range(len(arr)):
            start = max(0, i - half_window)
            end = min(len(arr), i + half_window + 1)
            window = arr[start:end]
            smoothed_arr.append(sum(window) / len(window))

        return smoothed_arr

    CETs = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3.csv')
    ID = list(set(CETs['ID']))
    CE_tree = [];
    CE_grass = [];
    CE_cropland = []
    for id in ID:
        try:
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
            CE_tree.append([id] + list(ce_tree))
            CE_grass.append([id] + list(ce_grass))
            CE_cropland.append([id] + list(ce_cropland))
            print(id)
        except ValueError:
            continue

    CE_tree = np.array(CE_tree)
    CE_grass = np.array(CE_grass)

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
    CE_tree_warm[:, 6:8] = CE_tree_warm[:, 6:8] - 0.5
    CE_grass_warm = np.array([row for row in CE_grass if row[0] in grass_warm_id])


    fig, ax = plt.subplots(1, 2, figsize=(11.5 * 0.7, 4.2 * 0.7))
    ET_warm = ET.loc[ET['ID'].isin(cities_warm['ID'])]
    ET_warm = ET_warm.iloc[:,1:-5].values/8*30*0.1
    ET_warm_rsp = np.reshape(ET_warm,(ET_warm.shape[0],7,46))
    ET_warm_arr = np.nanmean(ET_warm_rsp,axis=1)
    ET_warm_mean = np.nanmean(ET_warm_arr,axis=0)
    ax[1].plot(smooth_array(ET_warm_mean[::4],5),'-*',color='#d7191c',lw=2)

    tropical = climate.loc[climate['koppen'] == 1]
    ET_tropical = ET.loc[ET['ID'].isin(tropical['ID'])]
    ET_tropical = ET_tropical.iloc[:, 1:-5].values/8*30*0.1
    ET_tropical_rsp = np.reshape(ET_tropical, (ET_tropical.shape[0], 7, 46))
    ET_tropical_arr = np.nanmean(ET_tropical_rsp, axis=1)
    ET_tropical_mean = np.nanmean(ET_tropical_arr, axis=0)
    ax[1].plot(smooth_array(ET_tropical_mean[::4],5),'-*',color='#018571', lw=2)

    temperate = climate.loc[climate['koppen'] == 3]
    ET_temperate = ET.loc[ET['ID'].isin(temperate['ID'])]
    ET_temperate = ET_temperate.iloc[:, 1:-5].values/8*30*0.1
    ET_temperate_rsp = np.reshape(ET_temperate, (ET_temperate.shape[0], 7, 46))
    ET_temperate_arr = np.nanmean(ET_temperate_rsp, axis=1)
    ET_temperate_mean = np.nanmean(ET_temperate_arr, axis=0)
    ax[1].plot(smooth_array(ET_temperate_mean[::4],5),'-*',color='#8c510a', lw=2)
    ax[1].set_xticks([0, 3, 6, 9], ['Winter', 'Spring', 'Summer', 'Autumn'])
    ax[1].set_ylabel('ET (kg/m^2/month)')

    ax[0].plot(smooth_array(np.nanmean(CE_tree_tro[:, 1:], axis=0), 5), '-o', color='#018571', lw=2)
    ax[0].plot(smooth_array(np.nanmean(CE_tree_tem[:, 1:], axis=0), 5), '-o', color='#8c510a', lw=2)
    ax[0].plot(smooth_array(np.nanmean(CE_tree_warm[:,1:],axis=0),5),'-o',color='#d7191c',lw=2)
    ax[0].set_xticks([0,3,6,9],['Winter','Spring','Summer','Autumn'])
    ax[0].set_ylabel('TRC of urban trees (°C)')


    figToPath = (current_dir + '/4_Figures/FigS15_ET_CE_seasonal')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4,hspace=0.4)
    fig.savefig(figToPath, dpi=600)


