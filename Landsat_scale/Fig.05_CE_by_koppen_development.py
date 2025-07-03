import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib; matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
from matplotlib.pyplot import MultipleLocator

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv'
CETs = pd.read_csv(current_dir + relative_path)

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')
gdp_per = pd.read_csv(current_dir + '/2_Output/GDP_per_capita.csv')
world_countries = gpd.read_file(current_dir + '/2_Output/Shp/world_countries.shp')
world_countries = world_countries.rename(columns={'color_code':'Country Code'})
world_countries = pd.merge(world_countries,gdp_per,on='Country Code')

merged_df = pd.merge(CETs, climate, on='ID')
cities = pd.merge(cities, merged_df, on='ID')

Tree_TRC_mean = []
Tree_TRC_sd = []
grass_TRC_mean = []
grass_TRC_sd = []

for i in [1,3,4,2]:
    tropical_city = cities[cities['koppen']==i]
    points_in_polygons = gpd.sjoin(tropical_city, world_countries, how="left", predicate='intersects')
    tropical_developed = points_in_polygons[points_in_polygons['2018']>12375]
    tropical_developing = points_in_polygons[points_in_polygons['2018']<12375]
    Tree_TRC_mean.append([tropical_developed['CEts_tree'].mean(),tropical_developing['CEts_tree'].mean()])
    Tree_TRC_sd.append([tropical_developed['CEts_tree'].std(),tropical_developing['CEts_tree'].std()])

    grass_TRC_mean.append([tropical_developed['CEts_grass'].mean(), tropical_developing['CEts_grass'].mean()])
    grass_TRC_sd.append([tropical_developed['CEts_grass'].std(), tropical_developing['CEts_grass'].std()])

Tree_TRC_mean = np.array(Tree_TRC_mean)
Tree_TRC_sd = np.array(Tree_TRC_sd)*1.5
grass_TRC_mean = np.array(grass_TRC_mean)
grass_TRC_sd = np.array(grass_TRC_sd)*1.5

fig, ax = plt.subplots(1,2, figsize=(9*0.8,3), sharey=True)
x = np.array([0.9, 1.9,2.9,3.9])
ax[0].errorbar(x, Tree_TRC_mean[:,0], yerr=Tree_TRC_sd[:,0], fmt="o", mfc='none',ms=8, color='b',lw=2.5)
ax[0].errorbar(x+0.2, Tree_TRC_mean[:,1], yerr=Tree_TRC_sd[:,1], fmt="o", ms = 8, mfc='none', color='r',lw=2.5) # tropical # temporal

ax[1].errorbar(x, grass_TRC_mean[:,0], yerr=grass_TRC_sd[:,0], fmt="o", mfc='none', ms=8,color='b',lw=2.5)
ax[1].errorbar(x+0.2, grass_TRC_mean[:,1], yerr=grass_TRC_sd[:,1], fmt="o", ms=8, mfc='none', color='r',lw=2.5) # tropical # temporal

# ax.bar(x+1, merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].mean(), yerr=merged_df[merged_df['koppen']==4].iloc[:,[1,3,5]].std(), width=0.1,color=['#8582BD','#4F99C9','#A8D3A0']) # boreal

ax[0].set_xticks([1,2,3,4], ['Tropical', 'Temperate', 'Boreal', 'Arid'])
ax[1].set_xticks([1,2,3,4], ['Tropical', 'Temperate', 'Boreal', 'Arid'])
ax[0].set_ylabel('Tveg-Tbu (°C)')
#
figToPath = current_dir + '/4_Figures/Fig05_development_TRC.png'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)


