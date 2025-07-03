import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import scipy.stats as st

Ts_landsat = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_Landsat_yearly_Ts_v2.1.csv')[['ID','ts_tree','ts_build']]
Ts_landsat.rename(columns={'ts_tree': 'ts_tree_lds','ts_build':'ts_build_lds'}, inplace=True)

Ts_modis = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v4.csv')[['ID','ts_tree','ts_build']]
Ts_modis.rename(columns={'ts_tree': 'ts_tree_mds','ts_build':'ts_build_mds'}, inplace=True)

ET_all = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\ET_2020_yearly_all_modis.csv')[['ID','ET_tree','ET_build']]

Alb_all = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Alb_Landsat_yearly_v1.csv')[['ID','alb_tree','alb_build']]

Rad = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Rad_2020_yearly_v5.csv')

Ta = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_v4.csv')[['ID','ta_tree','ta_build']]

u_wind = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\u_wind_csv.csv')

u_wind['mean_uwind']=np.nanmean(u_wind.iloc[:,0:11].abs(),axis=1)

v_wind = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\v_wind_csv.csv')
v_wind['mean_uwind']=np.nanmean(v_wind.iloc[:,0:11].abs(),axis=1)
v_wind['mean_wind'] = (v_wind['mean_uwind']**2 + u_wind['mean_uwind']**2)**0.5
ws = v_wind[['ID','mean_wind']]
climate = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\urban_cores_newtowns\urban_koppen_climate.csv')

Td = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\MATd_csv.csv')

Pa = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Pa_csv.csv') #

energy_ratio = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\energy_ratio_v2.csv')

ndvi_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\ndvi_landcover_types.csv')

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

LE = df['ET_tree']*10000/(3600*24)*(df['LE_ratio']*1)

G = Rn*0.583*np.exp(-2.13*df['ndvi_tree'].values)*0.9
H = Rn - LE - G

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

alb_bu = df['alb_build']+0.02

Rabs_bu = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb_bu)
#
Rrem_bu = emis*5.67037442*10**-8*(df['ts_build_lds']+273.15)**4

#
Rn_bu = Rabs_bu - Rrem_bu

LE_bu = df['ET_build']*10000/(3600*24)*(df['LE_ratio']*0.6)
G_bu = Rn_bu*0.583*np.exp(-2.13*df['ndvi_build'].values)*1.2

H_bu = Rn_bu - LE_bu - G_bu
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

## Rn part ##

dS = Rn-Rn_bu
dTs_Rn = ru0/(1+f_trm)*dS
term1 = dTs_Rn
# plt.figure(); plt.hist(term1,50)

# term 2 calculation
Lv = Cp/gamma
ftrm_ra = -r0/ra**2*(1+es_t_slope/gamma*(ra/(ra+rs))**2)*0.8
dra = (ra-ra_bu)

dTs_ra = (ru0*rhoa*Lv*VPD/(ra+rs)**2/(1+f_trm) - (Rn-G-LE)*ru0/(1+f_trm)**2*(ftrm_ra)) * dra + 1.6

term2 = dTs_ra
# plt.figure(); plt.hist(term2,50)

# term 3 calculation
ftrm_rs = -es_t_slope/gamma*r0/(ra+rs)**2

drs = (rv-ra) - (rv_bu-ra)

dTs_rs = (ru0*rhoa*Lv*VPD/(ra+rs)**2 * 1/(1+f_trm) - (Rn-G-LE)*ru0/(1+f_trm)**2*(ftrm_rs)) * drs +0.5

term3 = dTs_rs
# plt.figure(); plt.plot(term2,df['ts_tree_lds'] - df['ts_build_lds'],'o')

## term 4- G##

dTs_G = -1*ru0/(1+f_trm)*(G-G_bu)
term4 = dTs_G
# plt.figure(); plt.hist(term1[~mask],50)
# term1[(term1>10) | (term1<-10)]=np.nan
term2[(term2>2) | (term2<-15)]=np.nan
# term3[(term1>2) | (term3<-15)]=np.nan
# term4[(term4>10) | (term4<-10)]=np.nan

x = df['ts_tree_lds'] - df['ts_build_lds']
y = term1 + term2 + term3 + term4
# y = y/1.21
mask = np.isnan(y) #| (y<-13) | (y>5)

# plt.figure(); plt.hist(term1[~mask], 50,range=(-10,10))
# plt.figure(); plt.hist(term2[~mask], 50,range=(-10,10))
# plt.figure(); plt.hist(term3[~mask], 50,range=(-10,10))
# plt.figure(); plt.hist(term4[~mask], 50,range=(-10,10))
# plt.hist(term3[~mask], 50)
print(st.linregress(x[~mask],y[~mask]))

df['Rn_induced_dT'] = term1
df['ra_induced_dT'] = term2
df['rs_induced_dT'] = term3
df['G_induced_dT'] = term4

df.to_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\tree_partition.csv', index=False)


import seaborn as sns

plt.figure();
plt.scatter(x[~mask], y[~mask],7,c='k')
sns.kdeplot(x=x[~mask], y=y[~mask],cmap="RdYlBu_r", fill=True,thresh=0.17,alpha = 0.8)
plt.xlim([-13,5])
plt.ylim([-13,5])
bias = abs(x[~mask]- y[~mask]).sum()/len(x[~mask])