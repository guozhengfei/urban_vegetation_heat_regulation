import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
import scipy.stats as st
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import os
import geopandas as gpd

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

relative_path4 = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_summer_normal.csv'
CETs_normal = pd.read_csv(current_dir+relative_path4)
CE_3d_v2 = np.load(current_dir+'/2_Output/Cooling_Efficiency/dT_vegetation_fraction_extreme.npy')

# dT = CE_3d_v2[0,:,:]
# slopes = (dT[:,3]-dT[:,-3])/95  # slope
# (slopes<0).sum()/731

slope = np.polyfit(np.linspace(0.97,0.3,95),CE_3d_v2[0,:,3:98].T,1)[0,:]
slope[slope>2.5]=2.5
slope2 = np.polyfit(np.linspace(0.97,0.3,95),CE_3d_v2[1,:,3:98].T,1)[0,:]
slope3 = np.polyfit(np.linspace(0.97,0.3,95),CE_3d_v2[2,:,3:98].T,1)[0,:]
slope3[slope3>5.5]=5.5

CETs_merge = pd.merge
CETs = CETs_normal.copy()

CETs['slope_tree'] = slope
CETs['slope_grass'] = slope2
CETs['slope_crop'] = slope3
tropical = gpd.read_file(current_dir + '/2_Output/Shp/koppen_tropical.shp')
arid = gpd.read_file(current_dir + '/2_Output/Shp/koppen_arid.shp')
temperate = gpd.read_file(current_dir + '/2_Output/Shp/koppen_temperate.shp')
boreal = gpd.read_file(current_dir + '/2_Output/Shp/koppen_boreal.shp')

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000
climate['koppen'][climate['koppen'] <= 3] = 1 # tropical
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2 # temperate
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3 # boreal
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4 # Arid
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5 # other


cities = gpd.read_file(current_dir + '/2_Output/Shp/points_citis.shp')
cities = pd.merge(cities, CETs, on='ID')
cities = pd.merge(cities, climate, on='ID')
fig, ax = plt.subplots(3,2,figsize=(10.5*0.72, 11.2*0.7),gridspec_kw={'width_ratios': [3, 1]})

tropical.plot(ax=ax[0,0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[0,0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[0,0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[0,0], color='lightgrey', edgecolor='none',alpha=0.4)

cities_cool = cities.loc[(cities['slope_tree'] < 0)]
cities_warm = cities.loc[(cities['slope_tree'] > 0)]
cities_warm.plot('slope_tree',ax=ax[0,0], marker='o',  markersize=6, cmap='Reds_r',vmin=0, vmax=4 )
cities_cool.plot('slope_tree',ax=ax[0,0], marker='o',  markersize=6, cmap='Blues',vmin=-3, vmax=0 )

ax[0,0].set_ylim([-54,80])
ax[0,0].set_xlim([-149,170])

def plot_slope_histogram(ax, cities_df, slope_col):
    """Helper function to create histogram of slope values with blue-to-red color mapping."""
    slopes = cities_df[slope_col].values
    
    # Create histogram
    n, bins, patches = ax.hist(slopes, bins=15, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color mapping from blue to red based on bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    # Normalize bin centers to [0, 1] for color mapping
    norm_centers = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    
    # Apply blue-to-red colormap
    import matplotlib.cm as cm
    colors = cm.RdBu_r(norm_centers)
    
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Slope')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

plot_slope_histogram(ax[0,1], cities, 'slope_tree')
ax[0,1].tick_params(labelsize=12)


### grassland
tropical.plot(ax=ax[1,0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[1,0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[1,0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[1,0], color='lightgrey', edgecolor='none',alpha=0.4)

cities_cool = cities.loc[(cities['slope_grass'] < 0)]
cities_warm = cities.loc[(cities['slope_grass'] > 0)]
cities_warm.plot('slope_grass',ax=ax[1,0], marker='o',  markersize=6, cmap='Reds_r',vmin=0, vmax=3 )
cities_cool.plot('slope_grass',ax=ax[1,0], marker='o',  markersize=6, cmap='Blues',vmin=-3, vmax=0 )

ax[1,0].set_ylim([-54,80])
ax[1,0].set_xlim([-149,170])

plot_slope_histogram(ax[1,1], cities, 'slope_grass')
ax[1,1].tick_params(labelsize=12)


### cropland
tropical.plot(ax=ax[2,0], color='#F78A5D', edgecolor='none',alpha=0.4)
temperate.plot(ax=ax[2,0], color='#AAD664', edgecolor='none',alpha=0.4)
arid.plot(ax=ax[2,0], color='#FFC96E', edgecolor='none',alpha=0.4)
boreal.plot(ax=ax[2,0], color='lightgrey', edgecolor='none',alpha=0.4)

cities_cool = cities.loc[(cities['slope_crop'] < 0)]
cities_warm = cities.loc[(cities['slope_crop'] > 0)]
cities_warm.plot('slope_crop',ax=ax[2,0], marker='o',  markersize=6, cmap='Reds_r',vmin=0, vmax=3 )
cities_cool.plot('slope_crop',ax=ax[2,0], marker='o',  markersize=6, cmap='Blues',vmin=-3, vmax=0 )
ax[2,0].set_ylim([-54,80])
ax[2,0].set_xlim([-149,170])

plot_slope_histogram(ax[2,1], cities, 'slope_crop')
ax[2,1].tick_params(labelsize=12)

plt.tight_layout()
figToPath = (current_dir + '/4_Figures/Fig.S11_dT_extreme_global_map.png')
os.makedirs(os.path.dirname(figToPath), exist_ok=True)
fig.savefig(figToPath, dpi=900)
plt.show()


