import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
import scipy.stats as st

CE_3d_v2 = np.load(r'D:\Projects\Postdoc urban greening\Data\Cooling_Efficiency\dT_vegetation_fraction_extreme.npy')

fig, axs = plt.subplots(1,3,figsize=(9,3),sharey=True)
treefraction = np.linspace(97,3,95)
dT0 = np.mean(CE_3d_v2[0,:,:],axis=0)
dT = dT0[3:-3]-(dT0[0]-dT0[3])
axs[0].plot(treefraction,dT,color='black',lw=2)

se = np.nanstd(CE_3d_v2[0,:,:],axis=0)[3:-3]/700**0.5*2
axs[0].fill_between(treefraction, dT - se, dT + se, color='grey', alpha=0.3, label='Standard Deviation')

grassfraction = np.linspace(97,3,95)
dT0 = np.mean(CE_3d_v2[1,:,:],axis=0)
dT = dT0[3:-3]-(dT0[0]-dT0[3])
axs[1].plot(treefraction,dT,color='black',lw=2)

se = np.nanstd(CE_3d_v2[1,:,:],axis=0)[3:-3]/700**0.5*2
axs[1].fill_between(grassfraction, dT - se, dT + se, color='grey', alpha=0.3, label='Standard Deviation')

cropfraction = np.linspace(97,3,95)
dT0 = np.mean(CE_3d_v2[2,:,:],axis=0)
dT = dT0[3:-3]-(dT0[0]-dT0[3])
axs[2].plot(treefraction,dT,color='black',lw=2)

se = np.nanstd(CE_3d_v2[2,:,:],axis=0)[3:-3]/700**0.5*2
axs[2].fill_between(cropfraction, dT - se, dT + se, color='grey', alpha=0.3, label='Standard Deviation')

figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03b_dT_vegetationfracion_landsat_extreme'
fig.tight_layout()
fig.savefig(figToPath, dpi=600)
plt.close(fig)

fig, axs = plt.subplots(1,3,figsize=(4.5,1.5),sharey=False)
dT = CE_3d_v2[0,:,:]
density = st.gaussian_kde(dT[:,3]-dT[:,-3])  # slope

n, x, _ = plt.hist(dT[:,3]-dT[:,-3], bins=np.linspace(-2.5, 2.5, 20),histtype=u'step', density=True)
axs[0].plot(x, density(x),'k')

dT = CE_3d_v2[1,:,:]
density = st.gaussian_kde(dT[:,3]-dT[:,-3])  # slope
n, x, _ = plt.hist(dT[:,3]-dT[:,-3], bins=np.linspace(-3, 3, 20),histtype=u'step', density=True)
axs[1].plot(x, density(x),'k')

dT = CE_3d_v2[2,:,:]
density = st.gaussian_kde(dT[:,3]-dT[:,-3])  # slope
n, x, _ = plt.hist(dT[:,3]-dT[:,-3], bins=np.linspace(-3, 3, 20),histtype=u'step', density=True)

axs[2].clear(); axs[2].plot(x, density(x),'k')

fig.tight_layout()

figToPath = r'D:\Projects\Postdoc urban greening\Figures_main\Fig03b_dT_vegetationfracion_hist_landsat_extreme'
fig.savefig(figToPath, dpi=600)