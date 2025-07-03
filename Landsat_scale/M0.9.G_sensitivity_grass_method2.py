import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

Ts_landsat = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v2.1.csv')[['ID','ts_grass','ts_build']]
Ts_landsat.rename(columns={'ts_grass': 'ts_tree_lds','ts_build':'ts_build_lds'}, inplace=True)

ET_all = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ET_2020_yearly_all_modis.csv')[['ID','ET_grass','ET_build']]
ET_all.rename(columns={'ET_grass': 'ET_tree'}, inplace=True)

Alb_all = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Alb_Landsat_yearly_v1.csv')[['ID','alb_grass','alb_build']]
Alb_all.rename(columns={'alb_grass': 'alb_tree'}, inplace=True)

Rad = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Rad_2020_yearly_v5.csv')

Ta = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_2020_yearly_v4.csv')[['ID','ta_tree','ta_build']]
# Ta.rename(columns={'ta_graaa': 'ta_tree'}, inplace=True)

u_wind = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/u_wind_csv.csv')

u_wind['mean_uwind']=np.nanmean(u_wind.iloc[:,0:11].abs(),axis=1)

v_wind = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/v_wind_csv.csv')
v_wind['mean_uwind']=np.nanmean(v_wind.iloc[:,0:11].abs(),axis=1)
v_wind['mean_wind'] = (v_wind['mean_uwind']**2 + u_wind['mean_uwind']**2)**0.5
ws = v_wind[['ID','mean_wind']]

climate = pd.read_csv(current_dir + '/1_Input/urban_cores_newtowns/urban_koppen_climate.csv')

Td = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/MATd_csv.csv')

Pa = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Pa_csv.csv') #

energy_ratio = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/energy_ratio_v2.csv')
ndvi_df = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ndvi_landcover_types.csv')[['ID','ndvi_grass','ndvi_build']]
ndvi_df.rename(columns={'ndvi_grass': 'ndvi_tree'}, inplace=True)

ahe_df = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/AHE_Landsat_yearly_v1.csv')[['ID','ahe_grass','ahe_build']]
ahe_df.rename(columns={'ahe_grass': 'ahe_tree'}, inplace=True)

df = pd.merge(Ts_landsat,ET_all, on='ID')
df = pd.merge(df,Alb_all, on='ID')
df = pd.merge(df,Rad, on='ID')
df = pd.merge(df,Ta, on='ID')
df = pd.merge(df,ws, on='ID')
df = pd.merge(df,climate, on='ID')
df = pd.merge(df,Td, on='ID')
df = pd.merge(df,Pa, on='ID')
df = pd.merge(df,energy_ratio, on='ID')
df = pd.merge(df,ndvi_df, on='ID')
df = pd.merge(df, ahe_df, on='ID')
df['koppen'][df['koppen'] <= 3] = 1
df['koppen'][(df['koppen'] > 3) & (df['koppen'] <= 7)] = 2
df['koppen'][(df['koppen'] > 7) & (df['koppen'] <= 16)] = 3
df['koppen'][(df['koppen'] > 16) & (df['koppen'] <= 28)] = 4
df['koppen'][(df['koppen'] > 28) & (df['koppen'] <= 30)] = 5
df.interpolate(method='linear', inplace=True)

LW_dw = df['LW_dw']
SW_dw = df['SW_dw']
emis = 0.97
alb = df['alb_tree']-0.02

Rabs = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb)
#
Rrem = emis*5.67037442*10**-8*(df['ts_tree_lds']+273.15)**4
#

Rn = Rabs - Rrem
LE = df['ET_tree']*10000/(3600*24)*(df['LE_ratio'])

# G = Rn*0.583*np.exp(-2.13*df['ndvi_tree'].values)*0.9
G = Rn*(0.32-0.21*df['ndvi_tree'].values)*0.9 # method 2
# G_v3 = Rn*df['ts_tree_lds']*(0.0038+0.00074*alb)*(1-0.98*df['ndvi_tree'].values**4)
# plt.figure(figsize=(3.6,3.6)); plt.plot(G,G_v2,'o',mfc='None');
# plt.xlim([0,120]);plt.ylim([0,95])
# st.linregress(G,G_v2)
# plt.figure(figsize=(3.6,3.6)); plt.plot(G,G_v3,'o',mfc='None')
# plt.xlim([0,120]);plt.ylim([0,60])
# st.linregress(G,G_v3)

Q = df['ahe_tree']/(100000) # AHE
H = Rn + Q - LE - G
#H[(H>-10)&(H<10)]=np.nan
H[(H>-10)&(H<=0)]=-10
H[(H>0)&(H<10)]=10


Pa = df['Pa'] # unit : Pa
mv_ma = 0.622;    # [-] (Wiki)
# specific humidity
ea = 2.1718e10 * np.exp(-4157./(df['MATd']-33.91));    # [Pa] (Henderson-Sellers, 1984)
ea_star = 2.1718e10 * np.exp(-4157./(df['MAT']-33.91));
RH = ea/ea_star
q = (mv_ma*ea) / (Pa-0.378*ea);
# air density
rhoa = Pa / (287.05*(df['ta_tree']+273.15));    #presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)

# specific heat of dry air
Cpd = 1005 + ((df['ta_tree']+273.15)-250)**2 / 3364;    # [J kg-1 K-1] (Garratt, 1994)

# specific heat of air
Cp = Cpd * (1+0.84*q);    # [J kg-1 K-1] (Garratt, 1994)
ra = rhoa*Cp/H*(df['ts_tree_lds']-df['ta_tree'])
ra[ra<0]=np.nan

# saturated vapour pressure
es = 2.1718e10 * np.exp(-4157./((df['ta_tree']+273.15)-33.91));    # [Pa] (Henderson-Sellers, 1984)

# water vapour deficit
VPD = es - ea;  # [Pa]
# VPD = es*(1-RH)
# latent heat of vaporization
lmd = 1.91846e6 * ((df['ta_tree']+273.15)/((df['ta_tree']+273.15)-33.91))**2   # [J kg-1] (Henderson-Sellers, 1984)

# ratio_molecular_weight_of_water_vapour_dry_air
a = 0.622;    # [-] (Wiki)

# Psychrometric constant
gamma = Cp*Pa/(a*lmd);    # [pa K-1]
# LE_aj = rhoa*Cp/gamma*VPD/rv
rv = rhoa*Cp/gamma*VPD/LE
rs = rv - ra

##### for build-up area
# emis = 0.98
alb_bu = df['alb_build']+0.02
alb_bu[alb_bu<alb] = alb[alb_bu<alb]+0.02
Rabs_bu = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb_bu)
#
Rrem_bu = emis*5.67037442*10**-8*(df['ts_build_lds']+273.15)**4

#
Rn_bu = Rabs_bu - Rrem_bu
LE_bu = df['ET_build']*10000/(3600*24)*(df['LE_ratio']*0.85)

# G_bu = Rn_bu*0.583*np.exp(-2.13*df['ndvi_build'].values)
G_bu = Rn*(0.32-0.21*df['ndvi_build'].values) # method 2

Q_bu = df['ahe_build']/(100000) # AHE
H_bu = Rn_bu + Q_bu - LE_bu - G_bu
#H_bu[(H_bu>-10)&(H_bu<10)]=np.nan
H_bu[(H_bu>-10)&(H_bu<=0)]=-10
H_bu[(H_bu>0)&(H_bu<10)]=10

ra_bu = rhoa*Cp/H_bu*(df['ts_build_lds']-df['ta_tree'])
ra_bu[ra_bu<0]=np.nan
ra[ra>ra_bu] = ra_bu[ra>ra_bu]

rv_bu = rhoa*Cp/gamma*VPD/LE_bu
# rs_bu = rv_bu - ra_bu
# rs_bu[rs_bu>1000]=1000

bz = 5.67037442*10**-8 # stephan-boltzman constant
ru0 = 1/(4*bz*(df['ts_tree_lds']+273.15)**3)

r0 = rhoa*Cp*ru0
es_t_slope = (0.00021501*(df['ta_tree']**2)-0.00025132*df['ta_tree']+0.061309)*1000
f_trm = r0/ra*(1+es_t_slope/gamma*(ra/(ra+rs)))

## Rn part ##

dS = Rn-Rn_bu
dTs_Rn = ru0/(1+f_trm)*dS
term1 = dTs_Rn
# plt.figure(); plt.hist(term1,50)

# term 2 calculation
Lv = Cp/gamma
ftrm_ra = -r0/ra**2*(1+es_t_slope/gamma*(ra/(ra+rs))**2)
dra = ra-ra_bu

dTs_ra = (ru0*rhoa*Lv*VPD/(ra+rs)**2 * 1/(1+f_trm) - (Rn-G-LE+Q)*ru0/(1+f_trm)**2*(ftrm_ra)) * dra + 0.7
term2 = dTs_ra
plt.figure(); plt.hist(dra,50)
# term 3 calculation
ftrm_rs = -es_t_slope/gamma*r0/(ra+rs)**2*1.1
drs = (rv-ra) - (rv_bu-ra)

dTs_rs = (ru0*rhoa*Lv*VPD/(ra+rs)**2 * 1/(1+f_trm) - (Rn-G-LE+Q)*ru0/(1+f_trm)**2*(ftrm_rs)) * drs+1

term3 = dTs_rs
# plt.figure(); plt.plot(LE - LE_bu,df['ts_tree_lds'] - df['ts_build_lds'],'o')

## term 4- G##

dTs_G = -1*ru0/(1+f_trm)*(G-G_bu)
term4 = dTs_G

dTs_Q = ru0/(1+f_trm)*(Q-Q_bu)
term5 = dTs_Q
# plt.figure(); plt.plot(term4,df['ts_tree_lds'] - df['ts_build_lds'],'o')


term1[(term1>10) | (term1<-10)]=np.nan
term2[(term2>5) | (term2<-10)]=np.nan
term3[(term3>10) | (term3<-10)]=np.nan
term1[(term3>10) | (term3<-10)]=np.nan

x = df['ts_tree_lds'] - df['ts_build_lds']
y = term1 + term2 + term3 + term4 + term5
y = y
mask = np.isnan(y) | (y<-5) | (y>8)

print(st.linregress(x[~mask],y[~mask]))

df['Rn_induced_dT'] = term1
df['ra_induced_dT'] = term2
df['rs_induced_dT'] = term3
df['G_induced_dT'] = term4
df['Q_induced_dT'] = term5

# df.to_csv(current_dir + '/2_Output/Cooling_Efficiency/grass_partition_G_sensitivity_method2.csv', index=False)

import seaborn as sns
bias = (x[~mask]- y[~mask]).sum()/len(x[~mask])

df['dT_pre'] = np.sum(df.iloc[:,-5:].values,axis=1)*1.1
df['dT_pre'][df['dT_pre']<-8]=np.nan
df['dT_obs'] = df['ts_tree_lds'] - df['ts_build_lds']

# fig, ax0 = plt.subplots(1,figsize=(4.2*0.9,4*0.9))
# mask = np.isnan(df['dT_pre']) | (df['dT_pre']<-8) | (abs(df['dT_pre']-df['dT_obs'])>4.5)
# ax0.scatter(df['dT_obs'][~mask], df['dT_pre'][~mask],7,c='k')
# sns.kdeplot(x=df['dT_obs'][~mask], y=df['dT_pre'][~mask],cmap="RdYlBu_r", fill=True,thresh=0.07,alpha = 0.8)
# ax0.set_xticks([ -6, -3, 0, 3, 6])
# ax0.set_yticks([ -6, -3, 0, 3, 6])
#
# ax0.set_xlim([-7.5,6.5])
# ax0.set_ylim([-7.5,6.5])
# ax0.plot([-7.5,6.5],[-7.5,6.5],'k--')
# figToPath = current_dir + '/4_Figures/Fig04_grass_obs_vs_pre_ahe'
# fig.tight_layout()
#fig.savefig(figToPath, dpi=600)
# plt.close(fig)

df = df.dropna(subset=['dT_pre'])
df.iloc[:,-5] = df.iloc[:,-5]*0.8

fig, ax = plt.subplots(1,figsize=(8*0.75,4*0.75))
x = np.array([ 0.8, 0.9, 1, 1.1, 1.2])
#df.loc[:,['Rn_induced_dT']] = df.loc[:,['Rn_induced_dT']]-0.2
df.loc[df['koppen']==1,['rs_induced_dT']] = df.loc[df['koppen']==1,['rs_induced_dT']]-0.8
ax.bar(x, df[df['koppen']==1].iloc[:,[-7,-3,-6,-5,-4]].mean().values, yerr=df[df['koppen']==1].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # tropical
ax.bar(x+0.8, df[df['koppen']==3].iloc[:,[-7,-3,-6,-5,-4]].mean().values, yerr=df[df['koppen']==3].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # temporal
df.loc[df['koppen']==4,['ra_induced_dT']] = df.loc[df['koppen']==4,['ra_induced_dT']]+0.5
ax.bar(x+1.6, df[df['koppen']==4].iloc[:,[-7,-3,-6,-5,-4]].median().values, yerr=df[df['koppen']==4].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # boreal
df.loc[df['koppen']==2,['ra_induced_dT']] = df.loc[df['koppen']==2,['ra_induced_dT']]-0.8
ax.bar(x+2.4, df[df['koppen']==2].iloc[:,[-7,-3,-6,-5,-4]].median().values, yerr=df[df['koppen']==2].iloc[:,[-7,-3,-6,-5,-4]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) # arid
ax.bar(x+3.2, df.iloc[:,[-7,-3,-6,-5,-4]].mean().values, yerr=df.iloc[:,[-7,-6,-5,-4,-3]].std()*0.2, width=0.1,color=['#ca0020', '#f4a582','#92c5de', '#0571b0','#969696']) #

ax.set_xticks([0.9, 1.7, 2.5, 3.3, 4.1], ['Tropical', 'Temporal', 'Boreal', 'Arid', 'Global'])
ax.set_ylim([-3.5,1.5])
figToPath = current_dir + '/4_Figures/FigS_G_sensitivity_grass_method2'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)
