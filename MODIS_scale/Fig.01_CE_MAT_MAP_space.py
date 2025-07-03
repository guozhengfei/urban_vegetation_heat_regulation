import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

CETs = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v4.3.csv') # v5,v4.2

climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cities = gpd.read_file(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\points_citis.shp')

cities = pd.merge(cities, CETs, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(2,2,figsize=(11.5, 6.5),gridspec_kw={'width_ratios': [3.5, 1]})

cities_cool = cities.loc[(cities['CEts_tree'] < 0)]
cities_warm = cities.loc[(cities['CEts_tree'] > 0)]

values = cities_cool['CEts_tree']*1  # Values used to determine point colors

world.plot(ax=ax[0,0], color='lightgray', edgecolor='black')
cities_cool.plot(ax=ax[0,0], marker='o', c=values, markersize=12, cmap='Blues_r', vmin=-8, vmax=2)
cities_warm.plot(ax=ax[0,0], marker='o', c=values, markersize=12,cmap='Reds',vmin=-2, vmax=4)
ax[0,0].set_ylim([-56,80])
ax[0,0].set_xlim([-170,170])

ax[0,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=12, c=cities_cool['CEts_tree'], cmap='Blues_r',vmin=-8, vmax=3)
ax[0,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=12, c=cities_warm['CEts_tree'],cmap='Reds', vmin=-1, vmax=4)

cities_cool = cities.loc[(cities['CEts_grass'] < 0)]
cities_warm = cities.loc[(cities['CEts_grass'] > 0)]
values = cities_cool['CEts_grass']*1  # Values used to determine point colors

world.plot(ax=ax[1,0], color='lightgray', edgecolor='black')
cities_cool.plot(ax=ax[1, 0], marker='o', markersize=12, c=values, cmap='Blues_r', vmin=-8, vmax=2)
cities_warm.plot(ax=ax[1,0], marker='o', markersize=12, c=values, cmap='Reds',vmin=-2, vmax=4)
ax[1,0].set_ylim([-56,80])
ax[1,0].set_xlim([-170,170])

ax[1,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=12, c=cities_cool['CEts_grass'], cmap='Blues_r',vmin=-8, vmax=3)
ax[1,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=12, c=cities_warm['CEts_grass'], cmap='Reds', vmin=-1, vmax=4)

figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig01_CE_MAT_MAP_v5'
fig.tight_layout()
fig.subplots_adjust(wspace = 0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)

