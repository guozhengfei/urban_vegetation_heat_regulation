import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
# import shap
import geopandas as gpd
from scipy.ndimage import convolve1d
import scipy.io
import seaborn as sns
from scipy.optimize import curve_fit

CE_types = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v4.csv')

CE_aboundance = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_vegetation_abundance.csv')

CE_types_r2 = CE_types['r']**2
CE_abund_r2 = CE_aboundance['r']**2

np.nanmean(CE_types_r2)

np.nanmean(CE_abund_r2)
