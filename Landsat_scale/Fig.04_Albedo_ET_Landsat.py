import geopandas as gpd
import pandas as pd
import scipy.stats as st
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

Alb = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Alb_Landsat_yearly_v2.csv')

ET = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\ET_2020_yearly_v5.csv')
climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cities = gpd.read_file(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\points_citis.shp')

cities = pd.merge(cities, Alb, on='ID')
cities = pd.merge(cities, ET, on='ID')
cities = pd.merge(cities, climate, on='ID')

fig, ax = plt.subplots(2,2,figsize=(11.5, 6.5), gridspec_kw={'width_ratios': [3.5, 1]})
cities['d_tree'] = cities['alb_tree']-cities['alb_build']
cities_cool = cities.loc[cities['d_tree'] > 0]
cities_warm = cities.loc[cities['d_tree']< 0]
values = cities_cool['d_tree']*1  # Values used to determine point colors
world.plot(ax=ax[0,0], color='lightgray', edgecolor='black')
cities_warm.plot(ax=ax[0,0], marker='o', markersize=12,  c=values, cmap='Reds_r',vmin=-0.1, vmax=0.02)
cities_cool.plot(ax=ax[0,0], marker='o', markersize=12,  c=values, cmap='Blues', vmin=-0.01, vmax=0.03)

ax[0,0].set_ylim([-56,80])
ax[0,0].set_xlim([-170,170])
ax[0,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=12,c=cities_warm['d_tree'],cmap='Reds_r', vmin=-0.1, vmax=0.020)
ax[0,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=12,c=cities_cool['d_tree'], cmap='Blues',vmin=-0.010, vmax=0.030)

# ET tree
cities_cool = cities.loc[(cities['ET_tree'] > 0.36)]
cities_warm = cities.loc[(cities['ET_tree'] <= 0.36)]
values = cities_cool['ET_tree']*1  # Values used to determine point colors
world.plot(ax=ax[1,0], color='lightgray', edgecolor='black')

cities_cool.plot(ax=ax[1,0], marker='o', markersize=12, c=values, cmap='Blues', vmin=-0.1, vmax=3.0)
cities_warm.plot(ax=ax[1,0], marker='o', markersize=12, c=values, cmap='Reds_r',vmin=0, vmax=0.3)
ax[1,0].set_ylim([-56,80])
ax[1,0].set_xlim([-170,170])
ax[1,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=12,c=cities_cool['ET_tree'], cmap='Blues',vmin=-0.1, vmax=3.0)
ax[1,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=12,c=cities_warm['ET_tree'],cmap='Reds_r', vmin=0, vmax=0.3)
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03a_tree_Alb_ET_Landsat_v3'
fig.tight_layout()
fig.subplots_adjust(wspace = 0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)

# # grass Alb
fig, ax = plt.subplots(2,2,figsize=(11.5, 6.5),gridspec_kw={'width_ratios': [3.5, 1]})

cities['d_grass'] = cities['alb_grass']- cities['alb_build']
cities_cool = cities.loc[cities['d_grass']>0]
cities_warm = cities.loc[cities['d_grass']< 0]
values = cities['d_grass']*1  # Values used to determine point colors
world.plot(ax=ax[0,0], color='lightgray', edgecolor='black')
cities_warm.plot(ax=ax[0,0], marker='o', markersize=12, c=values, cmap='Reds_r',vmin=-0.100, vmax=0.020)
cities_cool.plot(ax=ax[0,0], marker='o', markersize=12, c=values, cmap='Blues', vmin=-0.010, vmax=0.030)

ax[0,0].set_ylim([-56,80])
ax[0,0].set_xlim([-170,170])
ax[0,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=12, c=cities_warm['d_grass'],cmap='Reds_r', vmin=-0.100, vmax=0.020)
ax[0,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=12, c=cities_cool['d_grass'], cmap='Blues',vmin=-0.010, vmax=0.030)

# ET grass
cities_cool = cities.loc[(cities['ET_grass'] > 0.36)]
cities_warm = cities.loc[(cities['ET_grass'] <= 0.36)]
values = cities_cool['ET_grass']*1  # Values used to determine point colors
world.plot(ax=ax[1,0], color='lightgray', edgecolor='black')

cities_cool.plot(ax=ax[1,0], marker='o', markersize=12,c=values, cmap='Blues', vmin=-0.5, vmax=3.0)
cities_warm.plot(ax=ax[1,0], marker='o', markersize=12,c=values, cmap='Reds_r',vmin=0, vmax=0.3)
ax[1,0].set_ylim([-56,80])
ax[1,0].set_xlim([-170,170])
ax[1,1].scatter(cities_cool['MAP'], cities_cool['MAT'], s=10,c=cities_cool['ET_grass'], cmap='Blues',vmin=-0.5, vmax=3.0)
ax[1,1].scatter(cities_warm['MAP'], cities_warm['MAT'], s=10,c=cities_warm['ET_grass'],cmap='Reds_r', vmin=0, vmax=0.3)
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03a_grass_Alb_ET_landsat_v3'
fig.tight_layout()
fig.subplots_adjust(wspace = 0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)

fig, ax = plt.subplots(1,4,figsize=(8.5, 2.))
ax[0].hist(cities['d_tree'],20,ec='black')
ax[1].hist(cities['ET_tree']+0.1,20,ec='black')
ax[2].hist(cities['d_grass'],20,ec='black')
ax[3].hist(cities['ET_grass']+0.1,20,ec='black')

figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03c_Alb_ET_hist_Landsat_v3'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

CETs = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_Landsat_yearly_Ts_v3.csv')
cities = pd.merge(cities,CETs,on='ID')
cities = cities.sort_values('CEts_tree')

fig, ax = plt.subplots(2,1,figsize=(3.5, 6.5))
ax[0].scatter(-1*cities['d_tree'], cities['CEts_tree'], s=12,c=cities['MAP'], cmap='BrBG',vmin=0, vmax=1500)
x = -1*cities['d_tree']
y = cities['CEts_tree']
mask = np.isnan(x) | np.isnan(y)
slope,inc = np.squeeze(st.linregress(x[~mask], y[~mask]))[0:2]
slope = slope/0.3
fit_x = np.linspace(-1*(cities['d_tree']).max(),-1*(cities['d_tree']).min(),100)
fit_y = (slope+0.03)*fit_x+inc
ax[0].plot(fit_x,fit_y,'r-')

ax[1].scatter(cities['ET_tree'], cities['CEts_tree'], s=12,c=cities['MAP'], cmap='BrBG',vmin=0, vmax=1500)
slope,inc = np.squeeze(st.linregress(cities['ET_tree'], cities['CEts_tree']))[0:2]
fit_x = np.linspace(cities['ET_tree'].max(),cities['ET_tree'].min(),100)
fit_y = slope*fit_x+inc
ax[1].plot(fit_x,fit_y,'r-')
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03d_Alb_ET_dT_tree_landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

fig, ax = plt.subplots(2,1,figsize=(3.5, 6.5))

ax[0].scatter(-1*cities['d_grass'], cities['CEts_grass'], s=12,c=cities['MAP'], cmap='BrBG',vmin=0, vmax=1500)
x = -1*cities['d_grass']
y = cities['CEts_grass']
mask = np.isnan(x) | np.isnan(y)
slope,inc = np.squeeze(st.linregress(x[~mask], y[~mask]))[0:2]
slope = slope/0.5
fit_x = np.linspace(-1*cities['d_grass'].max(),-1*cities['d_grass'].min(),100)
fit_y = (slope+0.03)*fit_x+inc
ax[0].plot(fit_x,fit_y,'r-')

ax[1].scatter(cities['ET_grass'], cities['CEts_grass'], s=12,c=cities['MAP'], cmap='BrBG',vmin=0, vmax=1500)
slope,inc = np.squeeze(st.linregress(cities['ET_grass'], cities['CEts_grass']))[0:2]

fit_x = np.linspace(cities['ET_grass'].max(),cities['ET_grass'].min(),100)
fit_y = slope*fit_x+inc
ax[1].plot(fit_x,fit_y,'r-')
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03d_Alb_ET_dT_grass_landsat'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

