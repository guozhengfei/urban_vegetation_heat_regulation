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

Alb_all = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Alb_Landsat_yearly_v2.csv')[['ID','alb_tree','alb_build']]

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

energy_ratio = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\energy_ratio_csv.csv')

df = pd.merge(Ts_landsat,Ts_modis, on='ID')
df = pd.merge(df,ET_all, on='ID')
df = pd.merge(df,Alb_all, on='ID')
df = pd.merge(df,Rad, on='ID')
df = pd.merge(df,Ta, on='ID')
df = pd.merge(df,ws, on='ID')
df = pd.merge(df,climate, on='ID')
df = pd.merge(df,Td, on='ID')
df = pd.merge(df,Pa, on='ID')
df = pd.merge(df,energy_ratio, on='ID')

df.interpolate(method='linear', inplace=True)

LW_dw = df['LW_dw']
SW_dw = df['SW_dw']
emis = 0.97
alb = df['alb_tree']

Rabs = LW_dw*df['LW_ratio']*emis+SW_dw*(df['SW_ratio']*0.8)*(1-alb)
#
Rrem = emis*5.67037442*10**-8*(df['ts_tree_mds']+273.15)**4
#
Rn = Rabs - Rrem

LE = df['ET_tree']*10000/(3600*24)*(df['LE_ratio']*0.8)

G = Rn*0.583*np.exp(-2.13*0.75)
H = Rn - LE - G

Pa = df['Pa'] # unit : Pa
# ra = ra_cal(df['mean_wind'],30)
mv_ma = 0.622;    # [-] (Wiki)
ea = 0.611*np.exp(17.502*(df['MAT']-273.15)/(df['MAT']-273.15 +240.97))*1000; # [Pa] (Campbell and Norman., 2012)
# specific humidity
q = (mv_ma*ea) / (Pa-0.378*ea);
# air density
rhoa = Pa / (287.05*df['MAT']);    #presure unit: Pa; Ta unit: K; [kg m-3] (Garratt, 1994)

# specific heat of dry air
Cpd = 1005 + (df['MAT']-250)**2 / 3364;    # [J kg-1 K-1] (Garratt, 1994)

# specific heat of air
Cp = Cpd * (1+0.84*q);    # [J kg-1 K-1] (Garratt, 1994)
ra = abs(rhoa*Cp/H*(df['ts_tree_lds']-df['ta_tree']))

ea = 2.1718e10 * np.exp(-4157./(df['MATd']-33.91));    # [Pa] (Henderson-Sellers, 1984)

# saturated vapour pressure
es = 2.1718e10 * np.exp(-4157./(df['MAT']-33.91));    # [Pa] (Henderson-Sellers, 1984)

# water vapour deficit
VPD = es - ea;  # [Pa]

# latent heat of vaporization
lmd = 1.91846e6 * (df['MAT']/(df['MAT']-33.91))**2   # [J kg-1] (Henderson-Sellers, 1984)

# ratio_molecular_weight_of_water_vapour_dry_air
a = 0.622;    # [-] (Wiki)

# Psychrometric constant
gamma = Cp*Pa/(a*lmd);    # [pa K-1]
# LE_aj = rhoa*Cp/gamma*VPD/rv
rv = rhoa*Cp/gamma*VPD/LE
rs = rv - ra

##### for build-up area
alb_bu = df['alb_build']

Rabs_bu = LW_dw*df['LW_ratio']*emis+SW_dw*(df['SW_ratio']*0.8)*(1-alb_bu)
#
Rrem_bu = emis*5.67037442*10**-8*(df['ts_build_mds']+273.15)**4
#
Rn_bu = Rabs_bu - Rrem_bu

LE_bu = df['ET_build']*10000/(3600*24)*(df['LE_ratio']*0.3)

G_bu = Rn*0.583*np.exp(-2.13*0.3)

H_bu = Rn_bu - LE_bu - G_bu
ra_bu = abs(rhoa*Cp/H_bu*(df['ts_build_lds']-df['ta_tree']))


rv_bu = rhoa*Cp/gamma*VPD/LE_bu
rs_bu = rv_bu - ra_bu

## partation ##
bz = 5.67037442*10**-8 # stephan-boltzman constant
ru0 = 1/(4*bz*(df['ts_tree_mds']+273.15)**3)
bowen = H/LE
# plt.figure();plt.plot(H,LE,'o')
f = rhoa*Cp/(4*ra*bz*(df['ts_tree_mds']+273.15)**3)*(1+1/bowen)

dS = -1*(df['alb_tree']-df['alb_build'])*df['SW_dw']*(df['SW_ratio']*0.8)
term1 = ru0/(1+f)*dS

# plt.figure(); plt.hist(term1,50)

# term 2 calculation
dra = ra-ra_bu
df1 = -1*f*dra/ra

term2 = -1*ru0/(1+f)**2*Rn*df1
# plt.figure(); plt.hist(term2,50)
# term 3 calculation
dbowen = H/LE - H_bu/LE_bu
df2 = -1*rhoa*Cp/(4*ra*bz*(df['ts_tree_mds']+273.15)**3)*(dbowen/bowen**2)
term3 = -1*ru0/(1+f)**2*Rn*df2
plt.figure(); plt.hist(term3,50)

x = df['ts_tree_lds'] - df['ts_build_lds']
y = term1 + term2 + term3
plt.figure(); plt.plot(x,y,'o')
mask = (y>5) | (y<-15)
st.linregress(x[~mask],y[~mask])
plt.figure(); plt.plot(x[~mask], y[~mask], 'o')

# plt.figure(); plt.hist(term1[~mask],50)
plt.figure(); plt.hist(term2[~mask], 50)
plt.figure(); plt.hist(term3[~mask], 50)
# plt.hist(term3[~mask],50)