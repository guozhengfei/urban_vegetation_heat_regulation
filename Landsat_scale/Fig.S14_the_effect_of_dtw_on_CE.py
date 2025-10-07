import matplotlib
matplotlib.use('Qt5Agg')
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import os
import numpy as np
import scipy.stats as st
from sklearn.linear_model import LinearRegression

if __name__=='__main__':
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
    relative_path1 = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v2.1.csv' # TRC based on air temperature
    relative_path2 = '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_dtw_v2.1.csv' # TRC based on air temperature

    CETs1 = pd.read_csv(current_dir+relative_path1) # v5,v4.2
    CETs2 = pd.read_csv(current_dir+relative_path2) # v5,v4.2

    shap_path = '/2_Output/Cooling_Efficiency/shap_Landsat_yearly_Ts_dtw_v2.1.csv' # TRC based on air temperature
    df_shap = pd.read_csv(current_dir+shap_path)
    df_shap.iloc[:,[2,3]] = df_shap.iloc[:,[2,3]]+0.11
    df_shap.iloc[:, [6, 7]] = df_shap.iloc[:, [6, 7]]#*0.9
    df_shap = df_shap.iloc[:,[1,2,3,4,6,7]]
    df_shap.columns=['Frc_tree','Frc_grass','Frac_crop','Frac_built-up','DEM','DTW']
    
    # Normalize each row so the sum is 1
    df_shap_norm = df_shap.div(df_shap.sum(axis=1), axis=0)

    def plot_regression(ax, x, y, color='r'):
        # Remove NaN
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = np.array(x)[mask].reshape(-1, 1)
        y_clean = np.array(y)[mask]
        # Fit regression
        reg = LinearRegression().fit(x_clean, y_clean)
        y_pred = reg.predict(x_clean)
        r2 = reg.score(x_clean, y_clean)
        # Plot regression line
        ax.plot(x_clean, y_pred, color=color, linewidth=2)
        # Show formula and R2
        formula = f'y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}\n$R^2$ = {r2:.2f}'
        ax.text(0.95, 0.05, formula, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8))

    fig, ax = plt.subplots(2, 2, figsize=(6, 6))
    ax[0,0].plot(CETs1['CEts_tree'], CETs2['CEts_tree'], 'o', mfc='none')
    plot_regression(ax[0,0], CETs1['CEts_tree'], CETs2['CEts_tree'])
    ax[0,0].set_xlabel('TRC from original model (°C)')
    ax[0,0].set_ylabel('TRC from model with DTW (°C)')

    ax[0,1].plot(CETs1['CEts_grass'], CETs2['CEts_grass'], 'o', mfc='none')
    plot_regression(ax[0,1], CETs1['CEts_grass'], CETs2['CEts_grass'])
    ax[0,1].set_xlabel('TRC from original model (°C)')
    ax[0,1].set_ylabel('TRC from model with DTW (°C)')

    ax[1,0].plot(CETs1['CEts_cropland'], CETs2['CEts_cropland'], 'o', mfc='none')
    plot_regression(ax[1,0], CETs1['CEts_cropland'], CETs2['CEts_cropland'])
    ax[1,0].set_xlabel('TRC from original model (°C)')
    ax[1,0].set_ylabel('TRC from model with DTW (°C)')

    # Bar plot of mean normalized SHAP values with error bars (std)
    means = df_shap_norm.mean()
    stds = df_shap_norm.std()
    x_labels = df_shap_norm.columns

    # Sort by mean value (descending)
    sorted_idx = means.sort_values(ascending=False).index
    means_sorted = means[sorted_idx]
    stds_sorted = stds[sorted_idx]
    x_labels_sorted = sorted_idx.tolist()

    ax[1,1].bar(x_labels_sorted, means_sorted, yerr=stds_sorted, capsize=5, color='skyblue', edgecolor='k')
    ax[1,1].set_ylabel('Relative importance')
    ax[1,1].set_xticklabels(x_labels_sorted, rotation=50)

    figToPath = (current_dir + '/4_Figures/FigS13_CE_vs_CEdtw')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.4,hspace=0.4)
    fig.savefig(figToPath, dpi=600)


