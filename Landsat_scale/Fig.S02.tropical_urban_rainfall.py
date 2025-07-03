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

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')
    P = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/P_monthly_urban.csv')
    P_arr = P.iloc[:,1:-5].values
    P_months = []
    for month in range(12):
        ind = np.linspace(0,60,6).astype(int)+month
        P_i = np.nanmean(P_arr[:,ind],axis=1)
        P_months.append(P_i)
    P_months = np.array(P_months).T
    P_months = pd.DataFrame(P_months)
    P_months['ID'] = P['ID']

    cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')

    P_months = pd.merge(cities,P_months,on='ID', how = 'left')

    climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

    climate['koppen'][climate['koppen'] <= 3] = 1  # tropical
    climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2  # arid
    climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3  # temporal
    climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4  # boreal
    climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5  # polar

    P_months = pd.merge(P_months,climate,on='ID',how='left')
    tropical_P = P_months.loc[P_months['koppen'] == 1, :]

    P_sea = np.mean(tropical_P.iloc[:,3:15], axis=0)
    std = np.array(smooth_array(np.std(tropical_P.iloc[:, 3:15], axis=0), 9))*0.5
    fig, ax = plt.subplots(2, 1, figsize=(6.5*0.8, 6*0.8), sharex=True, sharey='row')
    ax[0].plot(smooth_array(P_sea,5),lw=3)
    ax[0].fill_between(np.linspace(0,11,12), P_sea - std, P_sea + std, color='grey', alpha=0.3,
                        label='Standard Deviation')

    # CETs
    CETs = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3.csv')
    CE_grass = CETs[['ID','CEts_grass']].astype(float).values
    id_tropical = list(climate['ID'][(climate['koppen'] == 1)].values)
    CE_grass_tro = np.array([row for row in CE_grass if row[0] in id_tropical])[:,1]
    CE_grass_tro_rsp = np.reshape(CE_grass_tro,(65,12))
    ce_mean = np.array(smooth_array(np.nanmean(CE_grass_tro_rsp,axis=0),5))
    ce_std = np.array(smooth_array(np.nanstd(CE_grass_tro_rsp,axis=0),5))*0.2
    ax[1].plot(ce_mean,'-',color='#018571',lw=3)
    ax[1].fill_between(np.linspace(0, 11, 12), ce_mean - ce_std, ce_mean + ce_std, color='grey', alpha=0.3,
                     label='Standard Deviation')

    figToPath = current_dir + '/4_Figures/FigS02_tropical_rainfall_seasonality'
    fig.tight_layout()
    # fig.subplots_adjust(wspace = 0.01)
    fig.savefig(figToPath, dpi=600)
    plt.close(fig)