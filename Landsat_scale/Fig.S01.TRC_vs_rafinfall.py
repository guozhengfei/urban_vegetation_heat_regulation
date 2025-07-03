import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import scipy.stats as st
import matplotlib; matplotlib.use('Qt5Agg')
import os

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v3.csv'
CETs = pd.read_csv(current_dir+relative_path) # v5,v4.2

climate = pd.read_csv(current_dir + '/2_Output/urban_koppen_climate.csv')
climate['MAT'] = climate['MAT']-273.15
climate['MAP'] = climate['MAP']*24*1000
df = pd.merge(CETs,climate, on='ID')


from scipy.optimize import curve_fit
import numpy as np
# Define the logarithmic function
def logarithmic_func(x, a, b):
    return a * np.log(x) + b

# Generating the fitted curve
X_fit = np.linspace(1, 10, 100)
Y_fit = logarithmic_func(X_fit, a, b)

fig, ax = plt.subplots(2,3,figsize=(9.5, 6.5))
x = df['MAP']; y = df['CEts_tree']
params, covariance = curve_fit(logarithmic_func, x, y)
a, b = params
ax[0,0].plot(x,y,'o',mfc='none')
min_x = x.min()
max_x = x.max()
X_fit = np.linspace(min_x, max_x, 20)
Y_fit = logarithmic_func(X_fit, a, b)
ax[0,0].plot(X_fit,Y_fit,'r-',lw=2)

y = df['CEts_grass']
params, covariance = curve_fit(logarithmic_func, x, y)
a, b = params
ax[0,1].plot(x,y,'o',mfc='none')
Y_fit = logarithmic_func(X_fit, a, b)
ax[0,1].plot(X_fit,Y_fit,'r-',lw=2)

y = df['CEts_cropland']
params, covariance = curve_fit(logarithmic_func, x, y)
a, b = params
ax[0,2].plot(x,y,'o',mfc='none')
Y_fit = logarithmic_func(X_fit, a, b)
ax[0,2].plot(X_fit,Y_fit,'r-',lw=2)

ax[1,0].plot(df['MAT'],df['CEts_tree'],'o',mfc='none')

ax[1,1].plot(df['MAT'],df['CEts_grass'],'o',mfc='none')

ax[1,2].plot(df['MAT'],df['CEts_cropland'],'o',mfc='none')


figToPath = (current_dir + '/4_Figures/Fig01s')
fig.tight_layout()
fig.subplots_adjust(wspace=0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)

