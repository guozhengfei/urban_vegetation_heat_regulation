import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\','/')

Ts_landsat = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_Landsat_yearly_Ts_v2.1.csv')[['ID','ts_tree','ts_build']]

Ts_landsat.rename(columns={'ts_tree': 'ts_tree_lds','ts_build':'ts_build_lds'}, inplace=True)

Ts_modis = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_2020_yearly_Ts_v4.csv')[['ID','ts_tree','ts_build']]
Ts_modis.rename(columns={'ts_tree': 'ts_tree_mds','ts_build':'ts_build_mds'}, inplace=True)

ET_all = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ET_2020_yearly_all_modis.csv')[['ID','ET_tree','ET_build']]

Alb_all = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Alb_Landsat_yearly_v1.csv')[['ID','alb_tree','alb_build']]

Rad = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/Rad_2020_yearly_v5.csv')

Ta = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/CE_2020_yearly_v4.csv')[['ID','ta_tree','ta_build']]

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

ndvi_df = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/ndvi_landcover_types.csv')

ahe_df = pd.read_csv(current_dir + '/2_Output/Cooling_Efficiency/AHE_Landsat_yearly_v1.csv')[['ID','ahe_tree','ahe_build']]
df = pd.merge(Ts_landsat,Ts_modis, on='ID')
df = pd.merge(df, ET_all, on='ID')
df = pd.merge(df, Alb_all, on='ID')
df = pd.merge(df, Rad, on='ID')
df = pd.merge(df, Ta, on='ID')
df = pd.merge(df, ws, on='ID')
df = pd.merge(df, climate, on='ID')
df = pd.merge(df, Td, on='ID')
df = pd.merge(df, Pa, on='ID')
df = pd.merge(df, energy_ratio, on='ID')
df = pd.merge(df, ndvi_df, on='ID')
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
alb = df['alb_tree']

Rabs = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb)
#
Rrem = emis*5.67037442*10**-8*(df['ts_tree_lds']+273.15)**4
#
Rn = Rabs - Rrem

LE = df['ET_tree']*10000/(3600*24)*(df['LE_ratio'])
# G = Rn*0.583*np.exp(-2.13*df['ndvi_tree'].values)*0.9
G = Rn*(0.32-0.21*df['ndvi_tree'].values)*0.9 # method 2
# G_v3 = Rn*df['ts_tree_lds']*(0.0038+0.00074*alb)*(1-0.98*df['ndvi_tree'].values**4)
#
# plt.figure(figsize=(3.6,3.6)); plt.plot(G,G_v2,'o',mfc='None')
# plt.xlim([0,120]);plt.ylim([0,95])
# st.linregress(G,G_v2)
#
# plt.figure(figsize=(3.6,3.6)); plt.plot(G,G_v3,'o',mfc='None')
# plt.xlim([0,120]);plt.ylim([0,65])
# st.linregress(G,G_v3)

Q = 0.1*df['ahe_tree']/(100000) # AHE
H = Rn + Q - LE - G
# plt.figure(); plt.hist(H,50)

H[(H>-5)&(H<5)]=np.nan
# plt.figure(); plt.plot(df['ts_tree_lds']-df['ts_build_lds'],G-G_bu,'o')

Pa = df['Pa'] # unit : Pa
mv_ma = 0.622;    # [-] (Wiki)
# specific humidity
ea = 2.1718e10 * np.exp(-4157./(df['MATd']-33.91));    # [Pa] (Henderson-Sellers, 1984)
ea_star = 2.1718e10 * np.exp(-4157./(df['MAT']-33.91));    # [Pa] (Henderson-Sellers, 1984)
RH_org = ea/ea_star
RH_mean = RH_org.mean()
RH = (RH_org-RH_mean)*1.1+RH_mean
# RH = ea/ea_star # 0-1
q = (mv_ma*ea) / (Pa-0.378*ea);
# air density
rhoa = Pa / (287.05*(df['ta_tree']+273.15));    #presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)

# specific heat of dry air
Cpd = 1005 + ((df['ta_tree']+273.15)-250)**2 / 3364;    # [J kg-1 K-1] (Garratt, 1994)

# specific heat of air
Cp = Cpd * (1+0.84*q);    # [J kg-1 K-1] (Garratt, 1994)
ra = rhoa*Cp/H*(df['ts_tree_lds']-df['ta_tree'])
ra[ra < 0] = np.nan
# plt.figure(); plt.hist(dra,50)

# saturated vapour pressure
es_org = 2.1718e10 * np.exp(-4157./((df['ta_tree']+273.15)-33.91))

# water vapour deficit
VPD = es_org - ea;  # [Pa]

# latent heat of vaporization
lmd = 1.91846e6 * ((df['ta_tree']+273.15)/((df['ta_tree']+273.15)-33.91))**2   # [J kg-1] (Henderson-Sellers, 1984)

# ratio_molecular_weight_of_water_vapour_dry_air
a = 0.622;    # [-] (Wiki)

# Psychrometric constant
gamma = Cp*Pa/(a*lmd);    # [pa K-1]
# LE_aj = rhoa*Cp/gamma*VPD/rv
rv = rhoa*Cp/gamma*VPD/LE
rs = rv - ra

alb_bu = df['alb_build']

Rabs_bu = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb_bu)
#
Rrem_bu = emis*5.67037442*10**-8*(df['ts_build_lds']+273.15)**4
#
Rn_bu = Rabs_bu - Rrem_bu

LE_bu = df['ET_build']*10000/(3600*24)*(df['LE_ratio']*0.8)
# G_bu = Rn_bu*0.583*np.exp(-2.13*df['ndvi_build'].values)
G_bu = Rn*(0.32-0.21*df['ndvi_build'].values) # method 2
# G_bu_v3 = Rn*df['ts_build_lds']*(0.0038+0.0074*alb)*(1-0.98*df['ndvi_build'].values**4)
#
# plt.figure(figsize=(3.6,3.6)); plt.plot(G_bu,G_bu_v2,'o',mfc='None')
# plt.xlim([0,150]);plt.ylim([0,120])
# st.linregress(G_bu,G_bu_v2)
# plt.figure(figsize=(3.6,3.6)); plt.plot(G_bu,G_bu_v3,'o',mfc='None')
# plt.xlim([0,150]);plt.ylim([0,85])
# st.linregress(G_bu,G_bu_v3)

Q_bu = df['ahe_build']/(100000) # AHE
H_bu = Rn_bu + Q_bu- LE_bu - G_bu
H_bu[(H_bu > -5) & (H_bu < 5)]=np.nan
ra_bu = rhoa*Cp/H_bu*(df['ts_build_lds']-df['ta_tree'])
ra_bu[ra_bu < 0] = np.nan

rv_bu = rhoa*Cp/gamma*VPD/LE_bu
# rs_bu = rv_bu - ra_bu
# rs_bu[rs_bu>2000] = 2000
# plt.figure(); plt.hist(rs_bu)
# np.nanpercentile(rs_bu,95)
bz = 5.67037442*10**-8 # stephan-boltzman constant
ru0 = 1/(4*emis*bz*(df['ts_tree_lds']+273.15)**3)
r0 = rhoa*Cp*ru0
es_t_slope = (0.00021501*(df['ta_tree']**2)-0.00025132*df['ta_tree']+0.061309)*1000 #[Pa k-1]
f_trm = r0/ra*(1+es_t_slope/gamma*(ra/(ra+rs)))

x = df['ts_tree_lds'] - df['ts_build_lds']

df['dT_obs'] = df['ts_tree_lds'] - df['ts_build_lds']
mask_obs = (df['dT_obs']>2) | (df['dT_obs']<-10)

fig, ax = plt.subplots(1, 5, figsize=(15*0.9, 3*0.9), sharey='row')
ax[0].plot((alb - alb_bu)[~mask_obs], df['dT_obs'][~mask_obs],'o', mfc='None')
coef = st.linregress(alb- alb_bu, df['dT_obs'])
maxV = np.percentile(alb - alb_bu,99)
minV = np.percentile(alb - alb_bu,1)
ax[0].plot([minV,maxV],[minV*coef.slope+coef.intercept,maxV*coef.slope+coef.intercept],'r-')

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
ax[2].plot([minV,maxV],[minV*coef.slope+coef.intercept,maxV*coef.slope+coef.intercept],'r-')

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
ax[4].set_ylim([-10,2])
figToPath = current_dir + '/4_Figures/FigS04_dT_flux_tree'
fig.tight_layout()
# fig.subplots_adjust(wspace = 0.01)
fig.savefig(figToPath, dpi=600)
plt.close(fig)