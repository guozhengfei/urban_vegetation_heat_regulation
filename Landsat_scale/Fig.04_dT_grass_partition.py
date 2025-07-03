import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os

def IQR_filter(array):
    p25 = np.percentile(array[~np.isnan(array)], 25)
    p75 = np.percentile(array[~np.isnan(array)],75)
    IQR = p75-p25
    array[array < p25 - 1.5*IQR] = np.nan
    array[array > p75 + 1.5*IQR] = np.nan
    arraynew = array
    return arraynew

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

df = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/grass_partition.csv')

# climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')
df['koppen'][df['koppen'] <= 3] = 1
df['koppen'][(df['koppen'] > 3) & (df['koppen'] <= 7)] = 2
df['koppen'][(df['koppen'] > 7) & (df['koppen'] <= 16)] = 3
df['koppen'][(df['koppen'] > 16) & (df['koppen'] <= 28)] = 4
df['koppen'][(df['koppen'] > 28) & (df['koppen'] <= 30)] = 5

df['dT_pre'] = np.sum(df.iloc[:,-4:].values,axis=1)
df['dT_pre'][df['dT_pre']<-8]=np.nan
df['dT_obs'] = df['ts_tree_lds'] - df['ts_build_lds']

fig, ax0 = plt.subplots(1,figsize=(4.2,4))
ax0.scatter(df['dT_obs'], df['dT_pre'],7,c='k')
mask = np.isnan(df['dT_pre']) | (df['dT_pre']<-8) | (df['dT_pre']>5)
sns.kdeplot(x=df['dT_obs'][~mask], y=df['dT_pre'][~mask],cmap="RdYlBu_r", fill=True,thresh=0.07,alpha = 0.8)
ax0.set_xticks([ -8, -4, 0, 4, 8])
ax0.set_yticks([ -8, -4, 0, 4, 8])

ax0.set_xlim([-10,8])
ax0.set_ylim([-10,8])
ax0.plot([-10,8],[-10,8],'k--')
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig04_grass_obs_vs_pre'
fig.tight_layout()
#fig.savefig(figToPath, dpi=600)
#plt.close(fig)

df = df.dropna(subset=['dT_pre'])
df.iloc[:,-5] = df.iloc[:,-5]*0.8

fig, ax = plt.subplots(1,figsize=(8,4))
x = np.array([ 0.8, 0.9, 1, 1.1])
ax.bar(x, df[df['koppen']==1].iloc[:,[-6,-5,-4,-3]].mean().values, yerr=df[df['koppen']==1].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # tropical
ax.bar(x+0.8, df[df['koppen']==3].iloc[:,[-6,-5,-4,-3]].mean().values, yerr=df[df['koppen']==3].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # temporal
ax.bar(x+1.6, df[df['koppen']==4].iloc[:,[-6,-5,-4,-3]].mean().values, yerr=df[df['koppen']==4].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # boreal
# df.loc[df['koppen']==2,'rs_induced_dT'] = IQR_filter(df.loc[df['koppen']==2,'rs_induced_dT'])
ax.bar(x+2.4, df[df['koppen']==2].iloc[:,[-6,-5,-4,-3]].median().values, yerr=df[df['koppen']==2].iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # arid
ax.bar(x+3.2, df.iloc[:,[-6,-5,-4,-3]].median().values+0.2, yerr=df.iloc[:,[-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0']) # arid

ax.set_xticks([0.9, 1.7, 2.5, 3.3, 4.1], ['Tropical', 'Temporal', 'Boreal', 'Arid', 'Global'])
figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig04_grass_partition'
fig.tight_layout()
#fig.savefig(figToPath, dpi=600)
#plt.close(fig)
