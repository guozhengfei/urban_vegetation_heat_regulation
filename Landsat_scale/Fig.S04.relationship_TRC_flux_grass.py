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

df['dT_obs'] = df['ts_tree_lds'] - df['ts_build_lds']
mask_obs = (df['dT_obs']>6) | (df['dT_obs']<-5)

fig, ax = plt.subplots(1, 5, figsize=(15*0.9, 3*0.9), sharey='row')
ax[0].plot((alb - alb_bu)[~mask_obs], df['dT_obs'][~mask_obs],'o', mfc='None')
coef = st.linregress(alb- alb_bu, df['dT_obs'])
maxV = np.percentile(alb - alb_bu,99)
minV = np.percentile(alb - alb_bu,1)
ax[0].plot([minV,maxV],[minV*2.6+coef.intercept,maxV*2.6+coef.intercept],'r-')

ax[1].plot((Q*0.1-Q_bu)[~mask_obs], df['dT_obs'][~mask_obs],'o', mfc='None')
coef = st.linregress(Q*0.1-Q_bu, df['dT_obs'])
maxV = np.percentile(Q*0.1-Q_bu,99)
minV = np.percentile(Q*0.1-Q_bu,1)
ax[1].plot([minV,maxV],[minV*coef.slope+coef.intercept,maxV*coef.slope+coef.intercept],'r--')
ax[1].set_xlim([-60,0])

ax[2].plot((H-H_bu)[~mask_obs], df['dT_obs'][~mask_obs],'o', mfc='None')
mask = np.isnan(H) | np.isnan(H_bu)
coef = st.linregress((H-H_bu)[~mask], df['dT_obs'][~mask])
maxV = np.nanpercentile(H-H_bu,99)
minV = np.nanpercentile(H-H_bu,1)
ax[2].plot([minV,maxV],[minV*coef.slope+coef.intercept,maxV*coef.slope+coef.intercept],'r--')

ax[3].plot((LE-LE_bu*0.1)[~mask_obs], df['dT_obs'][~mask_obs],'o', mfc='None')
coef = st.linregress(LE-LE_bu*0.1, df['dT_obs'])
maxV = np.percentile(LE-LE_bu*0.1,99)
minV = np.percentile(LE-LE_bu*0.1,1)
ax[3].plot([minV,maxV],[minV*coef.slope+coef.intercept,maxV*coef.slope+coef.intercept],'r-')

ax[4].plot((G-G_bu)[~mask_obs], df['dT_obs'][~mask_obs],'o', mfc='None')
coef = st.linregress(G-G_bu, df['dT_obs'])
maxV = np.percentile(G-G_bu,99)
minV = np.percentile(G-G_bu,1)
ax[4].plot([minV,maxV],[minV*coef.slope+coef.intercept,maxV*coef.slope+coef.intercept],'r-')
ax[4].set_ylim([-5,6])

figToPath = current_dir + '/4_Figures/FigS04_dT_flux_grass'
fig.tight_layout()
# fig.subplots_adjust(wspace = 0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)