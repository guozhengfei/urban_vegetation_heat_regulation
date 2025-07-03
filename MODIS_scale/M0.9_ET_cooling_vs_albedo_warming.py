import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from scipy import stats as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
# import shap
import geopandas as gpd
from scipy.ndimage import convolve1d
import scipy.io
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import curve_fit
from scipy import optimize

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def linear_fit(x, y):
    # Perform linear regression
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept

def log_fit(x, y):
    # Perform logarithmic regression
    params = np.polyfit(np.log(x), y, 1)
    slope = params[0]
    intercept = params[1]

    # Calculate predicted y-values
    y_pred = slope * np.log(x) + intercept

    # Calculate R-squared value
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calculate p-value
    p_value = st.linregress(np.log(x), y)[3]

    return slope, intercept, r_squared, p_value

CE = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_v4.csv')

CE_Ts = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v4.csv')

climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')

ET = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\ET_2020_yearly_v4.csv')

Alb = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\albedo_2020_yearly_v4.csv')

HG = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\HG_2020_yearly_v4.csv')

energy_ratios = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\energy_ratio_csv.csv')

Rn = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Rn_2020_yearly_v4.csv')
# FVC_max = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\FVC_2020_yearly_v4.csv')
# climate['koppen'][climate['koppen'] <= 3] = 1
# climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
# climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
# climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
# climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

merged_df = pd.merge(CE, climate, on='ID')
merged_df = pd.merge(merged_df, ET, on='ID')
merged_df = pd.merge(merged_df, Alb, on='ID')
merged_df = pd.merge(merged_df, CE_Ts, on='ID')
merged_df = pd.merge(merged_df, HG, on='ID')
merged_df = pd.merge(merged_df, energy_ratios, on='ID')
merged_df = pd.merge(merged_df, Rn.iloc[:,0:-1], on='ID')

# # Albedo_induced warming
energy_warm = -1*merged_df['Alb_cropland']*0.001*merged_df['SW']
energy_cool = merged_df['ET_cropland']*10**6*(2.501-2.361*0.001*merged_df['MAT'])
# st.linregress(energy_warm,merged_df['cropland'])
# st.linregress(energy_cool,merged_df['cropland'])
# st.linregress(energy_warm-energy_cool,merged_df['cropland'])
# plt.figure(); plt.plot(energy_warm-energy_cool,merged_df['cropland'],'o')
# plt.figure(); plt.plot(energy_warm,merged_df['cropland'], 'o')
# plt.figure(); plt.plot(energy_cool,merged_df['cropland'], 'o')
l0 = 1/(4*0.95*merged_df['MAT']**3*5.67037442*10**-8)
# plt.figure(); plt.hist(l0*pred,bins=20)


plt.figure(); plt.hist(merged_df['Alb_tree'],20,ec='black')
plt.figure(); plt.hist(merged_df['Alb_grass'],20,ec='black')
plt.figure(); plt.hist(merged_df['Alb_cropland'],20,ec='black')
np.sum(merged_df['Alb_tree'] > 0) / np.sum(~np.isnan(merged_df['Alb_tree']))

plt.figure(); plt.hist(merged_df['HG_tree'],20,ec='black')
plt.figure(); plt.hist(merged_df['HG_grass'],20,ec='black')
plt.figure(); plt.hist(merged_df['HG_cropland'],20,ec='black')
np.sum(merged_df['Alb_cropland'] < 0) / np.sum(~np.isnan(merged_df['Alb_cropland']))

plt.figure(); plt.hist(merged_df['ET_tree']+0.1,20,ec='black')
plt.figure(); plt.hist(merged_df['ET_grass']+0.1,20,ec='black')
plt.figure(); plt.hist(merged_df['ET_cropland']+0.1,20,ec='black')
np.sum(merged_df['Alb_cropland'] < 0) / np.sum(~np.isnan(merged_df['Alb_cropland']))


radiation_part = -1*merged_df['Alb_tree']*0.001*merged_df['SW']/(24*3600)*merged_df['SW_ratio']
ET_part = merged_df['ET_tree']*10**6*(2.501-2.361*0.001*(merged_df['MAT']-273.15))/(24*3600)#*merged_df['LW_ratio']

HG_part = merged_df['HG_tree']

pred = l0*(radiation_part-ET_part)

merged_df['CEts_tree'][merged_df['FVC_tree'] < 0.3]= np.nan
merged_df['ET_tree'][merged_df['ET_tree'] < -0.2 ]= np.nan
x = merged_df['ET_tree']
y = merged_df['CEts_tree']
mask = np.isnan(x) | np.isnan(y)
st.linregress(x[~mask]+0.1, y[~mask])
plt.figure(); plt.plot(merged_df['ET_tree']+0.1, merged_df['CEts_tree'],'o')

merged_df['CEts_grass'][merged_df['FVC_grass'] < 0.3]= np.nan
st.linregress(merged_df['ET_grass']+0.1, merged_df['CEts_grass'])
plt.figure(); plt.plot(merged_df['ET_grass']+0.1, merged_df['CEts_grass'],'o')

st.linregress(merged_df['ET_cropland']+0.1, merged_df['CEts_cropland'])
plt.figure(); plt.plot(merged_df['ET_cropland']+0.1, merged_df['CEts_cropland'],'o')

x = merged_df['Alb_tree']
y = merged_df['CEts_tree']
mask = np.isnan(x) | np.isnan(y)
st.linregress(x[~mask]+0.1, y[~mask])
st.linregress(merged_df['Alb_tree']*-1, merged_df['CEts_tree'])
plt.figure(); plt.plot(merged_df['Alb_tree']*-1, merged_df['CEts_tree'],'o')


x = merged_df['Alb_grass']
y = merged_df['CEts_grass']
mask = np.isnan(x) | np.isnan(y)
st.linregress(x[~mask]+0.1, y[~mask])
mask = np.isnan(x) | np.isnan(y)
st.linregress(merged_df['Alb_grass']*-1, merged_df['CEts_grass'])
plt.figure(); plt.plot(merged_df['Alb_grass']*-1, merged_df['CEts_grass'],'o')

st.linregress(merged_df['Alb_cropland']*-1, merged_df['CEts_cropland'])
plt.figure(); plt.plot(merged_df['Alb_cropland']*-1, merged_df['CEts_cropland'],'o')

st.linregress(merged_df['Rn_tree'], merged_df['CEts_tree'])
plt.figure(); plt.plot(merged_df['Rn_tree'], merged_df['CEts_tree'],'o')

st.linregress(merged_df['Rn_grass'], merged_df['CEts_grass'])
plt.figure(); plt.plot(merged_df['Rn_grass'], merged_df['CEts_grass'],'o')

LE_tree = merged_df['ET_tree']*10**6*(2.501-2.361*0.001*(merged_df['MAT']-273.15))/(3600*24)*2.8


st.linregress(merged_df['Rn_tree'], LE_tree)
plt.figure(); plt.plot(merged_df['Rn_tree'], LE_tree, 'o')

HG_tree = merged_df['Rn_tree']-LE_tree
st.linregress(HG_tree, merged_df['CEts_tree'])
plt.figure(); plt.plot(HG_tree, merged_df['CEts_tree'], 'o')
d_sw_up = -1*merged_df['Alb_tree']*0.001*merged_df['SW']/(24*3600)

st.linregress(d_sw_up, merged_df['CEts_tree'])

d_LE = merged_df['ET_tree']*10**6*2.45/(24*3600)
st.linregress(d_LE, merged_df['CEts_tree'])

st.linregress(l0*(d_sw_up - d_LE + HG_tree)*0.12, merged_df['CEts_tree'])
plt.figure(); plt.plot(l0*(d_sw_up - d_LE + HG_tree)*0.15, merged_df['CEts_tree'],'o')

plt.figure(); plt.plot(HG_tree, merged_df['CEts_tree'],'o')
merged_df['Rn_tree']-merged_df['ET_tree']*10**6*2.45/(24*3600)

from scipy.optimize import minimize
def objective(x):
    pred = l0 * (radiation_part - x[0]*ET_part + x[1]*HG_part)
    return abs(1-st.linregress(pred, merged_df['CEts_tree']).slope)
initial_guess = [0.1, 0.1]
result = minimize(objective, initial_guess)
x = result.x
pred = l0 * (radiation_part - x[0]*ET_part + x[1]*HG_part)
plt.figure(); plt.plot()


# obs = merged_df['CEts_tree'].values
# obs[obs<-10.9] = np.nan
# valid_indices = np.logical_and(~np.isnan(pred), ~np.isnan(obs))
# x_valid = obs[valid_indices]
# y_valid = pred[valid_indices]
# st.linregress(x_valid,y_valid)
# plt.figure(); plt.plot(x_valid,y_valid,'o')
st.linregress(pred, merged_df['CEts_tree'])
plt.figure(); plt.plot(pred, merged_df['CEts_tree'],'o')


st.linregress(l0*merged_df['HG_tree'], merged_df['CEts_tree'])
plt.figure(); plt.plot(merged_df['HG_tree'], merged_df['CEts_tree'], 'o')
st.linregress(energy_warm,merged_df['CEts_tree'])
st.linregress(merged_df['Alb_tree'],merged_df['CEts_tree'])
plt.figure(); plt.plot(merged_df['Alb_tree'],merged_df['CEts_tree'],'o')
st.linregress(energy_warm-energy_cool, merged_df['CEts_tree'])
plt.figure(); plt.plot(energy_warm-energy_cool, merged_df['CEts_tree'],'o')
plt.figure(); plt.plot(energy_warm, merged_df['CEts_tree'], 'o')
plt.figure(); plt.plot(energy_cool, merged_df['CEts_tree'], 'o')

# grass
st.linregress(merged_df['HG_grass'], merged_df['CEts_grass'])

energy_warm = -1*merged_df['Alb_grass']*0.001*merged_df['SW']/(3600*24)
energy_cool = merged_df['ET_grass']*10**6*(2.501-2.361*0.001*(merged_df['MAT']-273.15))/(3600*24)
HG_change = 0.95*5.67037442*10**-8*((merged_df['ts_grass']+273.15)**4-(merged_df['ts_build']+273.15)**4)
l0 = 1/(4*0.95*merged_df['MAT']**3*5.67037442*10**-8)
l0 = 1
st.linregress(HG_change*l0,merged_df['CEts_grass'])
st.linregress(energy_warm*l0,merged_df['CEts_grass'])
st.linregress(energy_cool*l0,merged_df['CEts_grass'])
st.linregress(energy_warm-energy_cool, merged_df['CEts_grass'])
plt.figure(); plt.plot(energy_warm-energy_cool,merged_df['CEts_grass'],'o')
plt.figure(); plt.plot(energy_warm,merged_df['CEts_grass'], 'o')
plt.figure(); plt.plot(energy_cool,merged_df['CEts_grass'], 'o')
plt.figure(); plt.plot(HG_change*l0,merged_df['CEts_grass'], 'o')

# crop
st.linregress(merged_df['HG_cropland'], merged_df['CEts_cropland'])

energy_warm = -1*merged_df['Alb_cropland']*0.001*merged_df['SW']
energy_cool = merged_df['ET_cropland']*10**6*(2.501-2.361*0.001*(merged_df['MAT']-273.15))
st.linregress(energy_warm,merged_df['CEts_cropland'])
st.linregress(energy_cool,merged_df['CEts_cropland'])
st.linregress(energy_warm-energy_cool,merged_df['CEts_cropland'])
plt.figure(); plt.plot(energy_warm-energy_cool,merged_df['CEts_cropland'],'o')
plt.figure(); plt.plot(energy_warm, merged_df['CEts_cropland'], 'o')
plt.figure(); plt.plot(energy_cool, merged_df['CEts_cropland'], 'o')
