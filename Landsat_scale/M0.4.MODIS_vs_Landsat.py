import pandas as pd
import scipy.stats as st
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

CETs_lad = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_Landsat_yearly_Ts_v3.csv')

CETs_mod = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_v1_Ts.csv')

df = pd.merge(CETs_lad,CETs_mod,on='ID')
fig, ax = plt.subplots(1,2, figsize=(7, 3.5), sharex=True, sharey=True)
df = df.loc[df['r_x']>0.5]
ax[0].plot(df['CEts_tree'],df['tree'],'o')
ax[0].set_xlim([-12,7])
ax[0].set_ylim([-12,7])
ax[0].plot([-12,7],[-12,7],'k--')
st.linregress(df['CEts_tree'],df['tree'])

ax[1].plot(df['CEts_grass'],df['grass'],'o')
ax[1].set_xlim([-12,7])
ax[1].set_ylim([-12,7])
ax[1].plot([-12,7],[-12,7],'k--')
st.linregress(df['CEts_grass'],df['grass'])

figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig_Modis_vs_Landsat'
fig.tight_layout()
# fig.subplots_adjust(wspace = 0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)
