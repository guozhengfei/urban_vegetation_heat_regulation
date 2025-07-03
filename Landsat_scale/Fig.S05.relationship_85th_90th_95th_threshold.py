import numpy as np
import scipy.stats as st
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import os

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
relative_path = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_summer_extreme_85th.csv'
CETs_85 = pd.read_csv(current_dir+relative_path)

relative_path2 = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_summer_extreme_90th.csv'
CETs_90 = pd.read_csv(current_dir+relative_path2)

relative_path3 = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_summer_extreme_95th.csv'
CETs_95 = pd.read_csv(current_dir+relative_path3)

fig, ax = plt.subplots(2,3,figsize=(10*0.7, 6.5*0.7))

ax[0,0].plot(CETs_90['CEts_tree'], CETs_85['CEts_tree'],'o',mfc='none')
coef = st.linregress(CETs_90['CEts_tree'], CETs_85['CEts_tree'])
min_x = CETs_90['CEts_tree'].min()
max_x = CETs_90['CEts_tree'].max()
ax[0,0].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

ax[0,1].plot(CETs_90['CEts_grass'], CETs_85['CEts_grass'],'o',mfc='none')
coef = st.linregress(CETs_90['CEts_grass'], CETs_85['CEts_grass'])
min_x = CETs_90['CEts_grass'].min()
max_x = CETs_90['CEts_grass'].max()
ax[0,1].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

ax[0,2].plot(CETs_90['CEts_cropland'], CETs_85['CEts_cropland'], 'o',mfc='none')
coef = st.linregress(CETs_90['CEts_cropland'], CETs_85['CEts_cropland'])
min_x = CETs_90['CEts_cropland'].min()
max_x = CETs_90['CEts_cropland'].max()
ax[0,2].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

ax[1,0].plot(CETs_90['CEts_tree'], CETs_95['CEts_tree'],'o',mfc='none')
coef = st.linregress(CETs_90['CEts_tree'], CETs_95['CEts_tree'])
min_x = CETs_90['CEts_tree'].min()
max_x = CETs_90['CEts_tree'].max()
ax[1,0].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

ax[1,1].plot(CETs_90['CEts_grass'], CETs_95['CEts_grass'],'o',mfc='none')
coef = st.linregress(CETs_90['CEts_grass'], CETs_95['CEts_grass'])
min_x = CETs_90['CEts_grass'].min()
max_x = CETs_90['CEts_grass'].max()
ax[1,1].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

ax[1,2].plot(CETs_90['CEts_cropland'], CETs_95['CEts_cropland'], 'o',mfc='none')
coef = st.linregress(CETs_90['CEts_cropland'], CETs_95['CEts_cropland'])
min_x = CETs_90['CEts_cropland'].min()
max_x = CETs_90['CEts_cropland'].max()
ax[1,2].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)
fig.tight_layout()
figToPath = (current_dir + '/4_Figures/FigS05_85th_90th_95th_TRC')
fig.savefig(figToPath, dpi=600)

relative_path4 = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_summer_normal.csv'
CETs_normal = pd.read_csv(current_dir+relative_path4)

fig, ax = plt.subplots(2,4,figsize=(13.5*0.7, 6.5*0.7))

x = CETs_90['ts_build']-CETs_normal['ts_build']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_85['ts_build']-CETs_normal['ts_build']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[0,0].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[0,0].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_tree']-CETs_normal['ts_tree']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_85['ts_tree']-CETs_normal['ts_tree']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[0,1].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[0,1].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_grass']-CETs_normal['ts_grass']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_85['ts_grass']-CETs_normal['ts_grass']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[0,2].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[0,2].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_cropland']-CETs_normal['ts_cropland']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_85['ts_cropland']-CETs_normal['ts_cropland']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[0,3].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[0,3].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_build']-CETs_normal['ts_build']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_95['ts_build']-CETs_normal['ts_build']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[1,0].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[1,0].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_tree']-CETs_normal['ts_tree']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_95['ts_tree']-CETs_normal['ts_tree']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[1,1].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[1,1].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_grass']-CETs_normal['ts_grass']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_95['ts_grass']-CETs_normal['ts_grass']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[1,2].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[1,2].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

x = CETs_90['ts_cropland']-CETs_normal['ts_cropland']
x[x>np.nanpercentile(x,95)]=np.nan
x[x<np.nanpercentile(x,5)]=np.nan
y = CETs_95['ts_cropland']-CETs_normal['ts_cropland']
y=y[~np.isnan(x)]
x=x[~np.isnan(x)]
ax[1,3].plot(x, y,'o',mfc='none')
coef = st.linregress(x, y)
min_x = x.min()
max_x = x.max()
ax[1,3].plot([min_x,max_x],[min_x*coef.slope+coef.intercept,max_x*coef.slope+coef.intercept],'r-',lw=2)

figToPath = (current_dir + '/4_Figures/FigS05_85th_90th_95th_temperature_increase')
fig.savefig(figToPath, dpi=600)

test = (CETs_90['ts_tree']-CETs_normal['ts_tree'])-(CETs_90['ts_build']-CETs_normal['ts_build'])
np.sum(test<0)/test.shape[0]

test2 = (CETs_90['ts_grass']-CETs_normal['ts_grass'])-(CETs_90['ts_build']-CETs_normal['ts_build'])
np.sum(test2<0)/test2.shape[0]

test3 = (CETs_90['ts_cropland']-CETs_normal['ts_cropland'])-(CETs_90['ts_build']-CETs_normal['ts_build'])
np.sum(test3<0)/test3.shape[0]