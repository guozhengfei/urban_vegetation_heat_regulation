import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)

import matplotlib; #matplotlib.use('Qt5Agg')
import os

current_dir = os.path.dirname(os.getcwd()).replace('\\','/')
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

cities_warm = cities.loc[cities['CEts_grass']>0]

fig, ax = plt.subplots(2,2,figsize=(11.5*0.7, 6.2*0.7),gridspec_kw={'width_ratios': [3.5, 1]})

cities_cool = cities.loc[(cities['CEts_tree'] < 0)]
cities_warm = cities.loc[(cities['CEts_tree'] > 0)]

# values = cities_cool['CEts_tree']*1  # Values used to determine point colors
# world.plot(ax=ax[0,0], color='none', edgecolor='black')
tropical.plot(ax=ax[0,0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[0,0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[0,0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[0,0], color='lightgrey', edgecolor='none',alpha=0.4)

cities_cool.plot('CEts_tree',ax=ax[0,0], marker='o',  markersize=6, cmap='Blues_r', vmin=-8, vmax=2)
cities_warm.plot('CEts_tree',ax=ax[0,0], marker='o', markersize=6,cmap='Reds',vmin=-2, vmax=4)
ax[0,0].set_ylim([-54,80])
ax[0,0].set_xlim([-149,170])

ax[0,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=6, c=cities_cool['CEts_tree'], cmap='Blues_r',vmin=-8, vmax=3)
ax[0,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=6, c=cities_warm['CEts_tree'],cmap='Reds', vmin=-1, vmax=4)
ax[0,1].tick_params(labelsize=12)

cities_cool = cities.loc[(cities['CEts_grass'] < 0)]
cities_warm = cities.loc[(cities['CEts_grass'] > 0)]
cities_warm = cities_warm.loc[cities_warm['MAP']<1200] # 1% outliers are removed
# values = cities_cool['CEts_grass']*1  # Values used to determine point colors

# world.plot(ax=ax[1,0], color='none', edgecolor='black')
tropical.plot(ax=ax[1,0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[1,0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[1,0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[1,0], color='lightgrey', edgecolor='none',alpha=0.4)
cities_warm.plot('CEts_grass',ax=ax[1,0], marker='o', markersize=6, cmap='Reds',vmin=-2, vmax=4)
cities_cool.plot('CEts_grass',ax=ax[1, 0], marker='o', markersize=6,  cmap='Blues_r', vmin=-8, vmax=2)
ax[1,0].set_ylim([-54,80])
ax[1,0].set_xlim([-149,170])
ax[1,1].tick_params(labelsize=12)

ax[1,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=6, c=cities_warm['CEts_grass'], cmap='Reds', vmin=-1, vmax=4)
ax[1,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=6, c=cities_cool['CEts_grass'], cmap='Blues_r',vmin=-8, vmax=3)


figToPath = (current_dir + '/4_Figures/Fig01_CE_MAT_MAP_landsat_v3')
fig.tight_layout()
fig.subplots_adjust(wspace=0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)

