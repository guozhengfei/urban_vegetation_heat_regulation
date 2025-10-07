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
G = Rn*0.583*np.exp(-2.13*df['ndvi_tree'].values)
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
ra = rhoa*Cp/H*(df['ts_tree_lds']-df['ta_tree'])*0.6
ra[ra<0]=np.nan

# saturated vapour pressure
es = 2.1718e10 * np.exp(-4157./((df['ta_tree']+273.15)-33.91));    # [Pa] (Henderson-Sellers, 1984)

# water vapour deficit
VPD = ea_star - ea;  # [Pa]
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

df['ra_grass'] = ra
df['rv_grass'] = rv

df_out = df[['ID','ra_grass','rv_grass']]
df_out.to_csv(current_dir + '/2_Output/Cooling_Efficiency/grass_ra_rs.csv', index=False)
