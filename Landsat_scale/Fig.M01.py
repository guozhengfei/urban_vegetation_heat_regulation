import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import os
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
relative_path = '/2_Output/Cooling_Efficiency/LST_time_sries.csv'
LSTs = pd.read_csv(current_dir+relative_path) # v5,v4.2

LSTs['Time'] = LSTs['Time'].str.replace('Jan','01').str.replace('Feb','02').str.replace('Mar','03').str.replace('Apr','04').str.replace('May','05').str.replace('Jun','06').str.replace('Jul','07').str.replace('Aug','08').str.replace('Sep','09').str.replace('Oct','10').str.replace('Nov','11').str.replace('Dec','12').str.replace(' ','-').str.replace(',','')

date_strings = LSTs['Time']  # Example date strings

date_series = []

for date_str in date_strings:
    date_obj = datetime.strptime(date_str, '%m-%d-%Y').date()
    date_series.append(date_obj)

LSTs['Time2'] = np.array(date_series)
LSTs['doy'] = pd.to_datetime(LSTs['Time2']).dt.dayofyear

plt.figure(); plt.plot(LSTs['doy'],LSTs['ST_B10'],'o',mfc='None')

summer_lst = LSTs[(LSTs['doy']>140) & (LSTs['doy']<245)]
summer_values = summer_lst['ST_B10'].values
plt.figure(); plt.plot(summer_values[~np.isnan(summer_values)],'o-',mfc='None')
p95 = np.percentile(summer_values[~np.isnan(summer_values)],90)
p05 = np.percentile(summer_values[~np.isnan(summer_values)],10)
plt.axhline(y=p95, color='red', linestyle='--')
plt.axhline(y=p05, color='red', linestyle='--')