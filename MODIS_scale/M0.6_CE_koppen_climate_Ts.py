import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
from scipy import stats as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
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

CE = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v4.csv')

climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')

climate['koppen'][climate['koppen'] <= 3] = 1
climate['koppen'][(climate['koppen'] > 3) & (climate['koppen'] <= 7)] = 2
climate['koppen'][(climate['koppen'] > 7) & (climate['koppen'] <= 16)] = 3
climate['koppen'][(climate['koppen'] > 16) & (climate['koppen'] <= 28)] = 4
climate['koppen'][(climate['koppen'] > 28) & (climate['koppen'] <= 30)] = 5

merged_df = pd.merge(CE, climate, on='ID')

# plot MAT vs Tree cooling
slope, intercept, r_squared, p_value = st.linregress(merged_df['MAT']-273.15, merged_df['CEts_tree'])[:4]
xd = np.linspace(merged_df['MAT'].min()-273.15,merged_df['MAT'].max()-273.15,100)
fit_line = slope * np.log(xd) + intercept

plt.figure()
plt.scatter(merged_df['MAT']-273.15, merged_df['CEts_tree'], label='Data')
plt.plot(xd, fit_line, 'r--', label='Fit Line')

# plot MAP vs Tree cooling
st.linregress(merged_df['MAP']*24, merged_df['CEts_tree'])
slope, intercept, r_squared, p_value = log_fit(merged_df['MAP']*24, merged_df['CEts_tree'])

st.linregress(merged_df['MAP']*24, merged_df['CEts_tree'])
slope, intercept, r_squared, p_value = log_fit(merged_df['MAP']*24, merged_df['CEts_tree'])
xd = np.linspace(merged_df['MAP'].min()*24,merged_df['MAP'].max()*24,100)
fit_line = slope * np.log(xd) + intercept

plt.figure()
plt.scatter(merged_df['MAP']*24, merged_df['CEts_tree'], label='Data')
plt.plot(xd, fit_line, 'r-', label='Fit Line')

# grass
# plot MAT vs Tree cooling
slope, intercept, r_squared, p_value = st.linregress(merged_df['MAT']-273.15, merged_df['CEts_grass'])[:4]
xd = np.linspace(merged_df['MAT'].min()-273.15,merged_df['MAT'].max()-273.15,100)
fit_line = slope * np.log(xd) + intercept

plt.figure()
plt.scatter(merged_df['MAT']-273.15, merged_df['CEts_grass'], label='Data')
plt.plot(xd, fit_line, 'r--', label='Fit Line')

# plot MAP vs Tree cooling
slope, intercept, r_squared, p_value = log_fit(merged_df['MAP']*24, merged_df['CEts_grass'])
xd = np.linspace(merged_df['MAP'].min()*24,merged_df['MAP'].max()*24,100)
fit_line = slope * np.log(xd) + intercept

plt.figure()
plt.scatter(merged_df['MAP']*24, merged_df['CEts_grass'], label='Data')
plt.plot(xd, fit_line, 'r-', label='Fit Line')

# cropland
slope, intercept, r_squared, p_value = st.linregress(merged_df['MAT']-273.15, merged_df['CEts_cropland'])[:4]
xd = np.linspace(merged_df['MAT'].min()-273.15,merged_df['MAT'].max()-273.15,100)
fit_line = slope * np.log(xd) + intercept

plt.figure()
plt.scatter(merged_df['MAT']-273.15, merged_df['CEts_cropland'], label='Data')
plt.plot(xd, fit_line, 'r--', label='Fit Line')

# plot MAP vs Tree cooling
slope, intercept, r_squared, p_value = log_fit(merged_df['MAP']*24, merged_df['CEts_cropland'])
xd = np.linspace(merged_df['MAP'].min()*24,merged_df['MAP'].max()*24,100)
fit_line = slope * np.log(xd) + intercept

plt.figure()
plt.scatter(merged_df['MAP']*24, merged_df['CEts_cropland'], label='Data')
plt.plot(xd, fit_line, 'r-', label='Fit Line')

# # tropical
# merged_df[merged_df['koppen'] == 1]['tree'].mean()
# merged_df[merged_df['koppen'] == 1]['grass'].mean()
# merged_df[merged_df['koppen'] == 1]['cropland'].mean()
#
# # arid
# merged_df[merged_df['koppen']==2]['tree'].mean()
# merged_df[merged_df['koppen']==2]['grass'].mean()
# merged_df[merged_df['koppen']==2]['cropland'].mean()
#
# # temporal
# merged_df[merged_df['koppen']==3]['tree'].mean()
# merged_df[merged_df['koppen']==3]['grass'].mean()
# merged_df[merged_df['koppen']==3]['cropland'].mean()
#
# # boreal
# merged_df[merged_df['koppen']==4]['tree'].mean()
# merged_df[merged_df['koppen']==4]['grass'].mean()
# merged_df[merged_df['koppen']==4]['cropland'].mean()
#
# import seaborn as sns
# plt.figure()
# plt.errorbar([0.9, 1, 1.1], merged_df[merged_df['koppen']==3].iloc[:,1:4].mean(), yerr=merged_df[merged_df['koppen']==3].iloc[:,1:4].std()*0.5, fmt='o', capsize=3)
#
# plt.errorbar([2.9, 3, 3.1], merged_df[merged_df['koppen']==1].iloc[:,1:4].mean(), yerr=merged_df[merged_df['koppen']==1].iloc[:,1:4].std()*0.5, fmt='o', capsize=3)
#
# plt.errorbar([1.9, 2, 2.1], merged_df[merged_df['koppen']==4].iloc[:,1:4].mean(), yerr=merged_df[merged_df['koppen']==4].iloc[:,1:4].std()*0.5, fmt='o', capsize=3)
#
# plt.errorbar([3.9, 4, 4.1], merged_df[merged_df['koppen']==2].iloc[:,1:4].mean(), yerr=merged_df[merged_df['koppen']==2].iloc[:,1:4].std()*0.5, fmt='o', capsize=3)
#
# sns.violinplot(data=merged_df[merged_df['koppen']==3].iloc[:,1:4],scale='count',order=['tree','grass','cropland'])
#
# sns.violinplot(data=merged_df[merged_df['koppen']==3].iloc[:,1:4],scale='count',order=['tree','grass','cropland'])
#
# # Set labels and title
# plt.xlabel('Data Sets')
# plt.ylabel('Value')