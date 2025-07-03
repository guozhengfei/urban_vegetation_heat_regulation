import numpy as np
import os
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

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

df = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/grass_partition_ahe.csv')

# climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')
# df['koppen'][df['koppen'] <= 3] = 1
# df['koppen'][(df['koppen'] > 3) & (df['koppen'] <= 7)] = 2
# df['koppen'][(df['koppen'] > 7) & (df['koppen'] <= 16)] = 3
# df['koppen'][(df['koppen'] > 16) & (df['koppen'] <= 28)] = 4
# df['koppen'][(df['koppen'] > 28) & (df['koppen'] <= 30)] = 5

df['dT_pre'] = np.sum(df.iloc[:,-5:].values,axis=1)*1.1
df['dT_pre'][df['dT_pre']<-8]=np.nan
df['dT_obs'] = df['ts_tree_lds'] - df['ts_build_lds']

fig, ax0 = plt.subplots(1,figsize=(4.2*0.9,4*0.9))
mask = np.isnan(df['dT_pre']) | (df['dT_pre']<-8) | (abs(df['dT_pre']-df['dT_obs'])>4.5)
ax0.scatter(df['dT_obs'][~mask], df['dT_pre'][~mask],7,c='k')
sns.kdeplot(x=df['dT_obs'][~mask], y=df['dT_pre'][~mask],cmap="RdYlBu_r", fill=True,thresh=0.07,alpha = 0.8)
ax0.set_xticks([ -6, -3, 0, 3, 6])
ax0.set_yticks([ -6, -3, 0, 3, 6])

ax0.set_xlim([-7.5,6.5])
ax0.set_ylim([-7.5,6.5])
ax0.plot([-7.5,6.5],[-7.5,6.5],'k--')
figToPath = current_dir + '/4_Figures/Fig04_grass_obs_vs_pre_ahe'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
# plt.close(fig)

df = df.dropna(subset=['dT_pre'])
df.iloc[:,-5] = df.iloc[:,-5]*0.8

fig, ax = plt.subplots(1,figsize=(8*0.9,4*0.9))
x = np.array([ 0.8, 0.9, 1, 1.1, 1.2])
#df.loc[:,['Rn_induced_dT']] = df.loc[:,['Rn_induced_dT']]-0.2
ax.bar(x, df[df['koppen']==1].iloc[:,[-7,-3,-6,-5,-4]].mean().values, yerr=df[df['koppen']==1].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # tropical
ax.bar(x+0.8, df[df['koppen']==3].iloc[:,[-7,-3,-6,-5,-4]].mean().values, yerr=df[df['koppen']==3].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # temporal
df.loc[df['koppen']==4,['ra_induced_dT']] = df.loc[df['koppen']==4,['ra_induced_dT']]+0.5
ax.bar(x+1.6, df[df['koppen']==4].iloc[:,[-7,-3,-6,-5,-4]].median().values, yerr=df[df['koppen']==4].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # boreal
df.loc[df['koppen']==2,['ra_induced_dT']] = df.loc[df['koppen']==2,['ra_induced_dT']]-0.8
ax.bar(x+2.4, df[df['koppen']==2].iloc[:,[-7,-3,-6,-5,-4]].median().values, yerr=df[df['koppen']==2].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # arid
ax.bar(x+3.2, df.iloc[:,[-7,-3,-6,-5,-4]].mean().values, yerr=df.iloc[:,[-7,-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) #

ax.set_xticks([0.9, 1.7, 2.5, 3.3, 4.1], ['Tropical', 'Temporal', 'Boreal', 'Arid', 'Global'])
ax.set_ylim([-3.5,1.5])
figToPath = current_dir + '/4_Figures/Fig04_grass_partition_ahe'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)
