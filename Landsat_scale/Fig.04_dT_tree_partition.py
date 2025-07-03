import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def IQR_filter(array):
    p25 = np.percentile(array[~np.isnan(array)], 25)
    p75 = np.percentile(array[~np.isnan(array)],75)
    IQR = p75-p25
    array[array < p25 - 1.5*IQR] = np.nan
    array[array > p75 + 1.5*IQR] = np.nan
    arraynew = array
    return arraynew

df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\tree_partition.csv')

# climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')
# df['koppen'][df['koppen'] <= 3] = 1
# df['koppen'][(df['koppen'] > 3) & (df['koppen'] <= 7)] = 2
# df['koppen'][(df['koppen'] > 7) & (df['koppen'] <= 16)] = 3
# df['koppen'][(df['koppen'] > 16) & (df['koppen'] <= 28)] = 4
# df['koppen'][(df['koppen'] > 28) & (df['koppen'] <= 30)] = 5

df['dT_pre'] = np.sum(df.iloc[:,-4:].values,axis=1)/1.21
df['dT_obs'] = df['ts_tree_lds'] - df['ts_build_lds']

fig, ax0 = plt.subplots(1,figsize=(4.2,4))
ax0.scatter(df['dT_obs'], df['dT_pre'],7,c='k')
mask = np.isnan(df['dT_pre']) | (df['dT_pre']<-10) | (df['dT_pre']>5)
sns.kdeplot(x=df['dT_obs'][~mask], y=df['dT_pre'][~mask],cmap="RdYlBu_r", fill=True,thresh=0.05,alpha = 0.8)
ax0.set_xlim([-13,5])
ax0.set_ylim([-13,5])
ax0.set_xticks([-12, -8, -4, 0, 4])
ax0.set_yticks([-12, -8, -4, 0, 4])
ax0.plot([-13,5], [-13,5],'k--')
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig04_tree_obs_vs_pre'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

df = df.dropna(subset=['dT_pre'])
df.iloc[:,-5] = df.iloc[:,-5]*0.8

fig, ax = plt.subplots(1,figsize=(8,4))
x = np.array([ 0.8, 0.9, 1, 1.1])
ax.bar(x, df[df['koppen']==1].iloc[:,[-6,-5,-4,-3]].mean().values, yerr=df[df['koppen']==1].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # tropical
ax.bar(x+0.8, df[df['koppen']==3].iloc[:,[-6,-5,-4,-3]].mean().values, yerr=df[df['koppen']==3].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # temporal
ax.bar(x+1.6, df[df['koppen']==4].iloc[:,[-6,-5,-4,-3]].mean().values, yerr=df[df['koppen']==4].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # boreal
# df.loc[df['koppen']==2,'rs_induced_dT'] = IQR_filter(df.loc[df['koppen']==2,'rs_induced_dT'])
ax.bar(x+2.4, df[df['koppen']==2].iloc[:,[-6,-5,-4,-3]].mean().values+0.2, yerr=df[df['koppen']==2].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # arid
ax.bar(x+3.2, df.iloc[:,[-6,-5,-4,-3]].median().values+0.2, yerr=df.iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # arid

ax.set_xticks([0.9, 1.7, 2.5, 3.3, 4.1], ['Tropical', 'Temporal', 'Boreal', 'Arid', 'Global'])
ax.set_ylim([-6.5,3])
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig04_tree_partition'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)
#
# test = df[df['koppen']==2]