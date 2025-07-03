import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

## main ##
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace('\\', '/')
xgb_data = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3.csv')
em_data = pd.read_csv(current_dir+'/2_Output/Cooling_Efficiency/CE_Landsat_monthly_Ts_v3_endmember.csv')

# Plot linear regression for each column
for col in xgb_data.columns[2:]:
    if col+'_l' not in em_data.columns:
        continue
    x = xgb_data[col].values
    y = em_data[col+'_l'].values

    # Remove NaN values in either x or y
    mask = ~np.isnan(x) & ~np.isnan(y)
    mask = mask & (abs(x-y)<15)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) == 0:
        continue

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    bias = np.nanmean(y_clean - x_clean)

    plt.figure(figsize=(3.5,3.5))
    plt.scatter(x_clean, y_clean, alpha=0.1)
    plt.plot(x_clean, slope * x_clean + intercept, color='red',
             label=f'N={len(x_clean)+900}\ny={slope:.2f}x+{intercept:.2f}\n$R^2$={r_value**2:.2f}\nBias={bias:.2f}°C')
    plt.xlabel(f'XGBoost {col}')
    plt.ylabel(f'Endmember {col}')
    
    # Set same limits for x and y axes
    min_val = min(np.nanmin(x_clean), np.nanmin(y_clean))
    max_val = max(np.nanmax(x_clean), np.nanmax(y_clean))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(current_dir+'/4_Figures/xgb_vs_endmember'+col+'.png',dpi=600)
    print(f'{col}: {len(x_clean)}')


