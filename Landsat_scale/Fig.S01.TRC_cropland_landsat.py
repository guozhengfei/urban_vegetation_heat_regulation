import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)

import matplotlib; matplotlib.use('Qt5Agg')
import os

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv'
CETs = pd.read_csv(current_dir+relative_path) # v5,v4.2

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
tropical = gpd.read_file(current_dir + '/2_Output/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/2_Output/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/2_Output/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/2_Output/Shp/koppen_boreal.shp')

cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')
cities = pd.merge(cities, CETs, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(1,2,figsize=(11.5, 3.5),gridspec_kw={'width_ratios': [3.1, 1]})

cities_cool = cities.loc[(cities['CEts_cropland'] < 0)]
cities_warm = cities.loc[(cities['CEts_grass'] > 0)]
cities_warm = cities_warm.loc[cities_warm['MAP']<1200] # 1% are removed
# values = cities_cool['CEts_grass']*1  # Values used to determine point colors

# world.plot(ax=ax[1,0], color='none', edgecolor='black')
tropical.plot(ax=ax[0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[0], color='lightgrey', edgecolor='none',alpha=0.4)
cities_warm.plot('CEts_cropland',ax=ax[0], marker='o', markersize=12, cmap='Reds',vmin=-2, vmax=4)
cities_cool.plot('CEts_cropland',ax=ax[0], marker='o', markersize=12,  cmap='Blues_r', vmin=-8, vmax=2)
ax[0].set_ylim([-56,80])
ax[0].set_xlim([-170,170])
ax[0].tick_params(labelsize=12)

ax[1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=12, c=cities_warm['CEts_cropland'], cmap='Reds', vmin=-1, vmax=4)
ax[1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=12, c=cities_cool['CEts_cropland'], cmap='Blues_r',vmin=-8, vmax=3)
ax[1].tick_params(labelsize=12)

figToPath = (current_dir + '/4_Figures/FigS01_TRC_cropland_landsat')
fig.tight_layout()
fig.subplots_adjust(wspace=0.01)
fig.savefig(figToPath, dpi=600)
# plt.close(fig)

