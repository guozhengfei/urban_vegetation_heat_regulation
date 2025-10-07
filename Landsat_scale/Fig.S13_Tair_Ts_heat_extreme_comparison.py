import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st
import pandas as pd
from matplotlib import pyplot as plt
plt.rc('font',family='Arial')
plt.tick_params(width=0.8,labelsize=14)
import os
import numpy as np

def identify_extreme_events(series, percentile=97.5):
    """Identifies extreme events in a time series based on a percentile threshold."""
    threshold = np.percentile(series.dropna(), percentile)
    return series > threshold

if __name__=='__main__':
    # --- IMPORTANT ---
    # Please update these file paths to your actual data files.
    # The CSVs should have a date column as the first column (or index)
    # and subsequent columns for each site's time series data.
    current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')
    TAIR_CSV_PATH = current_dir + '/2_Output/tmmx_csv_urban_751.csv'
    LST_CSV_PATH = current_dir + '/2_Output/lst_csv_urban_751.csv'
    tair_df = pd.read_csv(TAIR_CSV_PATH, index_col=0, parse_dates=True).iloc[:,:-5]*0.1
    lst_df = pd.read_csv(LST_CSV_PATH, index_col=0, parse_dates=True).iloc[:,:-5]/0.02-273.15

    # --- 3. Identify Extreme Events ---
    tair_extremes = tair_df.T.apply(identify_extreme_events).T
    lst_extremes = lst_df.T.apply(identify_extreme_events).T


    # Calculate the number of summer months with extreme events for each dataset
    tair_extreme_count = tair_extremes.values.sum()
    lst_extreme_count = lst_extremes.values.sum()

    # Find where both datasets agree on an extreme event
    agreement = (tair_extremes.values == lst_extremes.values).sum()

    # Calculate the percentage of agreement
    # This is the number of agreed extremes divided by the number of extremes identified by LST
    agreement_percentage = agreement / (lst_extremes.shape[0]*lst_extremes.shape[1])* 98

    # --- 5. Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # Left subplot: Scatter plot
    x = lst_df.values.reshape(-1)
    x[x>60]=np.nan
    y = tair_df.values.reshape(-1)
    mask = np.isnan(x)| np.isnan(y)
    axes[0].scatter(x[~mask], y[~mask], 1, alpha=0.1, color='grey')
    
    # Perform linear regression and plot the regression line
    slope, intercept, r_value, p_value, std_err = st.linregress(x[~mask], y[~mask])
    x_line = np.array([np.nanmin(x), np.nanmax(x)])
    y_line = slope * x_line + intercept
    axes[0].plot(x_line, y_line, 'r', label=f'y={slope:.2f}x+{intercept:.2f}\nR={r_value:.2f}')

    axes[0].set_ylabel('Tair (°C)', fontsize=12)
    axes[0].set_xlabel('LST (°C)', fontsize=12)
    axes[0].grid(True)
    axes[0].legend(loc='upper left', fontsize=12)

    # Right subplot: Pie chart
    labels = 'Agreement', 'Disagreement'
    sizes = [agreement_percentage, 100 - agreement_percentage]
    wedges, texts, autotexts = axes[1].pie(sizes, autopct='%1.1f%%', startangle=90, 
                                           colors=['lightblue','lightcoral'], textprops={'fontsize': 10})
    axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend

    # Save the figure
    output_filename = current_dir+'/4_Figures/Tair_LST_correlation_and_agreement.png'
    plt.savefig(output_filename, dpi=600)
    print(f"Figure saved to {output_filename}")
    plt.show()
    