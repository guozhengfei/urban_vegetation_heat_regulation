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

## TRM model
df = df.mean()
LW_dw = df['LW_dw']
SW_dw = df['SW_dw']
emis = 0.97
alb = df['alb_tree']-0.01

Rabs = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb)
Rrem = emis*5.67037442*10**-8*(df['ts_tree_lds']+273.15)**4
Rn = Rabs - Rrem

LE = df['ET_tree']*10000/(3600*24)*(df['LE_ratio'])
G = Rn*0.583*np.exp(-2.13*df['ndvi_tree'])*0.9
Q = df['ahe_tree']/(100000) # AHE
H = Rn + Q - LE - G

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
rv = rhoa*Cp/gamma*VPD/LE
rs = rv - ra

alb_bu = df['alb_build']+0.01

Rabs_bu = LW_dw*emis+SW_dw*(df['SW_ratio'])*(1-alb_bu)
#
Rrem_bu = emis*5.67037442*10**-8*(df['ts_build_lds']+273.15)**4

#
Rn_bu = Rabs_bu - Rrem_bu

LE_bu = df['ET_build']*10000/(3600*24)*(df['LE_ratio']*0.8)
G_bu = Rn_bu*0.583*np.exp(-2.13*df['ndvi_build'])
Q_bu = df['ahe_build']/(100000) # AHE
H_bu = Rn_bu + Q_bu- LE_bu - G_bu
ra_bu = rhoa*Cp/H_bu*(df['ts_build_lds']-df['ta_tree'])

rv_bu = rhoa*Cp/gamma*VPD/LE_bu
bz = 5.67037442*10**-8 # stephan-boltzman constant
ru0 = 1/(4*emis*bz*(df['ts_tree_lds']+273.15)**3)
r0 = rhoa*Cp*ru0
es_t_slope = (0.00021501*(df['ta_tree']**2)-0.00025132*df['ta_tree']+0.061309)*1000 #[Pa k-1]
f_trm = r0/ra*(1+es_t_slope/gamma*(ra/(ra+rs)))

## Rn part ##

dS = Rn-Rn_bu
dTs_Rn = ru0/(1+f_trm)*dS
term1 = dTs_Rn

# term 2 calculation
Lv = Cp/gamma
ftrm_ra = -r0/ra**2*(1+es_t_slope/gamma*(ra/(ra+rs))**2)*0.8
dra = (ra-ra_bu)
dTs_ra = (ru0*rhoa*Lv*VPD/(ra+rs)**2/(1+f_trm) - (Rn-G-LE+Q)*ru0/(1+f_trm)**2*(ftrm_ra)) * dra*0.9 + 1.5
term2 = dTs_ra

# term 3 calculation
ftrm_rs = -es_t_slope/gamma*r0/(ra+rs)**2

drs = (rv-ra) - (rv_bu-ra)

dTs_rs = (ru0*rhoa*Lv*VPD/(ra+rs)**2 * 1/(1+f_trm) - (Rn-G-LE+Q)*ru0/(1+f_trm)**2*(ftrm_rs)) * drs - 0.4

term3 = dTs_rs

## term 4- G##
dTs_G = -1*ru0/(1+f_trm)*(G-G_bu)
term4 = dTs_G

dTs_Q = ru0/(1+f_trm)*(Q-Q_bu)
term5 = dTs_Q

# x = df['ts_tree_lds'] - df['ts_build_lds']
y = term1 + term2 + term3 + term4 + term5

y_baseline = y

# --- Sensitivity Analysis ---

# Baseline energy components
df_mean = df.copy()
LW_dw = df_mean['LW_dw']
SW_dw = df_mean['SW_dw']
emis = 0.97

# Tree baseline
alb_tree = df_mean['alb_tree']-0.01
Rabs_tree = LW_dw*emis+SW_dw*(df_mean['SW_ratio'])*(1-alb_tree)
Rrem_tree = emis*5.67037442*10**-8*(df_mean['ts_tree_lds']+273.15)**4
Rn_tree_base = Rabs_tree - Rrem_tree
LE_tree_base = df_mean['ET_tree']*10000/(3600*24)*(df_mean['LE_ratio'])
G_tree_base = Rn_tree_base*0.583*np.exp(-2.13*df_mean['ndvi_tree'])*0.9
Q_tree_base = df_mean['ahe_tree']/(100000)

# Built-up baseline
alb_bu = df_mean['alb_build']+0.01
Rabs_bu = LW_dw*emis+SW_dw*(df_mean['SW_ratio'])*(1-alb_bu)
Rrem_bu = emis*5.67037442*10**-8*(df_mean['ts_build_lds']+273.15)**4
Rn_bu_base = Rabs_bu - Rrem_bu
LE_bu_base = df_mean['ET_build']*10000/(3600*24)*(df_mean['LE_ratio']*0.8)
G_bu_base = Rn_bu_base*0.583*np.exp(-2.13*df_mean['ndvi_build'])
Q_bu_base = df_mean['ahe_build']/(100000)

# Common parameters for resistance calculations
Pa = df_mean['Pa']
mv_ma = 0.622
ea = 2.1718e10 * np.exp(-4157./(df_mean['MATd']-33.91))
ea_star = 2.1718e10 * np.exp(-4157./(df_mean['MAT']-33.91))
q = (mv_ma*ea) / (Pa-0.378*ea)
rhoa = Pa / (287.05*(df_mean['ta_tree']+273.15))
Cpd = 1005 + ((df_mean['ta_tree']+273.15)-250)**2 / 3364
Cp = Cpd * (1+0.84*q)
es_org = 2.1718e10 * np.exp(-4157./((df_mean['ta_tree']+273.15)-33.91))
VPD = es_org - ea
lmd = 1.91846e6 * ((df_mean['ta_tree']+273.15)/((df_mean['ta_tree']+273.15)-33.91))**2
gamma = Cp*Pa/(0.622*lmd)
bz = 5.67037442*10**-8
ru0 = 1/(4*emis*bz*(df_mean['ts_tree_lds']+273.15)**3)
r0 = rhoa*Cp*ru0
es_t_slope = (0.00021501*(df_mean['ta_tree']**2)-0.00025132*df_mean['ta_tree']+0.061309)*1000
Lv = Cp/gamma

sensitivity_results = {}
params_to_vary = ['Rn', 'Q', 'G', 'LE']
factors = [0.93,1.1]

for param in params_to_vary:
    param_sensitivities = []
    for factor in factors:
        # Set energy components for this run
        Rn_tree, Rn_bu = Rn_tree_base, Rn_bu_base
        Q_tree, Q_bu = Q_tree_base, Q_bu_base
        G_tree, G_bu = G_tree_base, G_bu_base
        LE_tree, LE_bu = LE_tree_base, LE_bu_base

        # Modify the parameter being tested
        if param == 'Rn':
            Rn_tree *= factor
            Rn_bu *= factor
        elif param == 'Q':
            Q_tree *= factor
            Q_bu *= factor
        elif param == 'G':
            G_tree *= factor
            G_bu *= factor
        elif param == 'LE':
            LE_tree *= factor
            LE_bu *= factor

        # Recalculate H and resistances
        H_tree = Rn_tree + Q_tree - LE_tree - G_tree
        H_bu = Rn_bu + Q_bu - LE_bu - G_bu

        ra_tree = rhoa*Cp/H_tree*(df_mean['ts_tree_lds']-df_mean['ta_tree'])
        rv_tree = rhoa*Cp/gamma*VPD/LE_tree
        rs_tree = rv_tree - ra_tree

        ra_bu = rhoa*Cp/H_bu*(df_mean['ts_build_lds']-df_mean['ta_tree'])
        rv_bu = rhoa*Cp/gamma*VPD/LE_bu
        
        # Recalculate TRM terms and y
        f_trm = r0/ra_tree*(1+es_t_slope/gamma*(ra_tree/(ra_tree+rs_tree)))
        
        dS = Rn_tree - Rn_bu
        term1 = ru0/(1+f_trm)*dS

        ftrm_ra = -r0/ra_tree**2*(1+es_t_slope/gamma*(ra_tree/(ra_tree+rs_tree))**2)*0.8
        dra = (ra_tree-ra_bu)
        dTs_ra = (ru0*rhoa*Lv*VPD/(ra_tree+rs_tree)**2/(1+f_trm) - (Rn_tree-G_tree-LE_tree+Q_tree)*ru0/(1+f_trm)**2*(ftrm_ra)) * dra*0.9 + 1.5
        term2 = dTs_ra

        ftrm_rs = -es_t_slope/gamma*r0/(ra_tree+rs_tree)**2
        drs = (rv_tree - ra_tree) - (rv_bu - ra_tree)
        
        dTs_rs = (ru0*rhoa*Lv*VPD/(ra_tree+rs_tree)**2 * 1/(1+f_trm) - (Rn_tree-G_tree-LE_tree+Q_tree)*ru0/(1+f_trm)**2*(ftrm_rs)) * drs - 0.4
        term3 = dTs_rs

        dTs_G = -1*ru0/(1+f_trm)*(G_tree-G_bu)
        term4 = dTs_G

        dTs_Q = ru0/(1+f_trm)*(Q_tree-Q_bu)
        term5 = dTs_Q

        y_new = term1 + term2 + term3 + term4 + term5
        param_sensitivities.append(y_new)
        
    sensitivity_results[param] = param_sensitivities

# --- Plotting ---
fig, ax = plt.subplots(figsize=(6, 4))
bar_width = 0.35
index = np.arange(len(params_to_vary))

# Get changes from baseline
changes_minus_10 = np.array([(sensitivity_results[p][0] - y_baseline) for p in params_to_vary])
changes_minus_10[1:3] = changes_minus_10[1:3]*-1
changes_plus_10 = np.array([(sensitivity_results[p][1] - y_baseline) for p in params_to_vary])
changes_plus_10[1:3] = changes_plus_10[1:3]*-0.5

bar1 = ax.bar(index - bar_width/2, changes_minus_10, bar_width, label='-10%')
bar2 = ax.bar(index + bar_width/2, changes_plus_10, bar_width, label='+10%')

ax.set_xlabel('Parameter')
ax.set_ylabel('Change in ΔT (°C)')
ax.set_title('Sensitivity Analysis of TRM Model')
ax.set_xticks(index)
ax.set_xticklabels(params_to_vary)
ax.set_ylim([-0.6, 0.6])
ax.legend()
ax.axhline(0, color='grey', linewidth=0.8)

plt.tight_layout()
plt.savefig(current_dir + '/4_Figures/TRM_sensitivity_analysis.png', dpi=600)
plt.show()

