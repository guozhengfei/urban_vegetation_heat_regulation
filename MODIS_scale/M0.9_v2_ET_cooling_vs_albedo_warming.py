import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')

def replace_nan_with_nearest(arr):
    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(arr))[0]

    # Create linear interpolation function
    interp_func = interp1d(non_nan_indices, arr[non_nan_indices], fill_value='extrapolate')

    # Interpolate NaN values
    interpolated_values = interp_func(np.arange(len(arr)))

    return interpolated_values

Ta_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\Ta_max_csv.csv')
Ts_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\Ts_max_csv.csv')
H_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\H_max_csv.csv')
LE_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\LE_max_csv.csv')
Rn_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\net_radiation_csv.csv')
SW_down_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\SW_down_csv.csv')
wind_u_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\u_wind_csv.csv')
wind_v_df = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Energy\v_wind_csv.csv')
wind_org = ((wind_u_df.iloc[:,:-1]**2).values + (wind_v_df.iloc[:,:-1]**2).values)**0.5

Ta = replace_nan_with_nearest(np.mean(Ta_df.iloc[:,:-1],axis=1))
Ts = replace_nan_with_nearest(np.mean(Ts_df.iloc[:,:-1],axis=1))
H = replace_nan_with_nearest(np.mean(H_df.iloc[:,:-1],axis=1))/3600
LE = replace_nan_with_nearest(np.mean(LE_df.iloc[:,:-1],axis=1))/3600
Rn = replace_nan_with_nearest(np.mean(Rn_df.iloc[:,:-1],axis=1))/3600
SW_down = replace_nan_with_nearest(np.mean(SW_down_df.iloc[:,:-1].values,axis=1))/3600
wind = replace_nan_with_nearest(np.mean(wind_org,axis=1))

id = SW_down_df.iloc[:,-1]
df_energy = np.column_stack((id,Ta,Ts,H,LE,Rn,SW_down,wind))
df_energy = pd.DataFrame(df_energy)
df_energy.columns = ['ID','Ta','Ts','H','LE','Rn','SW_down','wind']
Alb = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\albedo_2020_yearly_v4.csv')
CETs = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\CE_2020_yearly_Ts_v4.csv')

df = pd.merge(CETs,Alb,on='ID')
df = pd.merge(df,df_energy,on='ID')

def ra_cal(WS,hc):
    # hc = 30 # canopy height; build-up:30m; tree:10m; grass:0.3m
    # WS = 2
    k = 0.4;    # von Kármán constant
    z0 = hc * 0.1;    # (Norman)
    ustar = WS*k/(np.log(10./z0))
    ra = WS/ustar**2+2/(k*ustar)#    % Eq. (2-4) in Ryu et al 2008
    ra[ra>100] = 100;
    return ra
Ps = 100*1000 # atmosphere pressure (Pa)

# ratio molecular weight of water vapour dry air
mv_ma = 0.622;    # [-] (Wiki)
ea = 0.611*np.exp(17.502*(df['Ta']-273.15)/(df['Ta']-273.15 +240.97))*1000; # [Pa] (Campbell and Norman., 2012)
# specific humidity
q = (mv_ma*ea) / (Ps-0.378*ea);
# air density
rhoa = Ps / (287.05*df['Ta']);    # [kg m-3] (Garratt, 1994)

# specific heat of dry air
Cpd = 1005 + (df['Ta']-250)**2 / 3364;    # [J kg-1 K-1] (Garratt, 1994)

# specific heat of air
Cp = Cpd * (1+0.84*q);    # [J kg-1 K-1] (Garratt, 1994)

# ratio_molecular_weight_of_water_vapour_dry_air

bz = 5.67037442*10**-8 # stephan-boltzman constant
ru0 = 1/(4*bz*df['Ts']**3)
bowen = df['H']/df['LE']
# plt.figure();plt.plot(H,LE,'o')
f=rhoa*Cp/(4*ra_cal(df['wind'],15)*bz*df['Ts']**3)*(1+1/bowen)

dS = -1*df['Alb_tree']/1000*df['SW_down']
term1 = ru0/(1+f)*dS

plt.figure(); plt.plot(df['CEts_tree'],term1,'o')
plt.figure(); plt.plot(df['CEts_tree'],df['ET_tree'],'o')
st.linregress(df['CEts_tree'],df['ET_tree'])

# term 2 calculation
dra = ra_cal(df['wind'],15)-ra_cal(df['wind'],10)
df1 = -1*f*dra/ra_cal(df['wind'],15)

term2 = -1*ru0/(1+f)**2*df['Rn']*df1
plt.figure(); plt.plot(df['CEts_tree'],term2,'o')
st.linregress(df['CEts_tree'],term2)

# term3 calculation

Rn2 = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\Rn_2020_yearly_v4.1.csv')
LE2 = pd.read_csv(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\ET_2020_yearly_v4.1.csv')
df = pd.merge(df,Rn2,on='ID')
df = pd.merge(df,LE2.iloc[:,:-1],on='ID')
plt.figure(); plt.plot(df['Rn'],(df['Rn_tree']+df['Rn_building']),'o')
st.linregress(df['Rn'],(df['Rn_tree']+df['Rn_building']))
plt.figure(); plt.plot(df['LE'],(df['ET_tree']+df['ET_building'])*2.45*10**6/(24*3600)*3, 'o')

H = Rn-LE
dbowen = H
df2 = rhoa*Cp/(4*ra_cal(df['wind'],30)*bz*df['Ts']**3)*(dbowen/bowen**2)
