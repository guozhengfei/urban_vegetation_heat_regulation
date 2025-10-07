import os
import numpy as np
import  rasterio
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
# Open the GeoTIFF file

LST_1km_files = os.listdir(r'D:\Projects\Postdoc urban greening\Data\TsMax')

LST_100m_files = os.listdir(r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\LST')

input_src = rasterio.open(r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\AHE\AHE_2010_year.tif')

for name in LST_1km_files:
    index = name.split('_')[-1]
    src_1km = rasterio.open(r'D:\Projects\Postdoc urban greening\Data\TsMax\\'+name)
    src_100m = rasterio.open(
        r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\LST\\' + 'Landsat_' + index)
    data_100m = src_100m.read()
    transform = src_100m.transform

    data_1km = src_1km.read()
    data_1km = data_1km[-85:-1,:,:]

    scale_factor_y = data_100m.shape[1] / data_1km.shape[1]
    scale_factor_x = data_100m.shape[2] / data_1km.shape[2]

    data_1km_nons = []
    for month in range(12):
        mon_ind = np.linspace(0,72,7).astype(int)+month
        data_1km_mon = np.nanmean(data_1km[mon_ind,:,:],axis=0)
        data_1km_mon[np.isnan(data_1km_mon)] = np.nanpercentile(data_1km_mon.reshape(-1),0.5)
        resampled_data = zoom(data_1km_mon, (scale_factor_y, scale_factor_x), order=1)
        data_1km_nons.append(resampled_data)
    data_1km_nons = np.array(data_1km_nons)

plt.figure(); plt.imshow(data_100m[6,:,:])

plt.figure(); plt.imshow(data_1km_nons[6,:,:])

plt.figure();plt.plot(data_100m[6,:,:].reshape(-1),data_1km_nons[6,:,:].reshape(-1),'.')
# transform = reference_src.transform
# output_profile = reference_src.profile
# output_profile.update(count=1, dtype='float32')
# filename = r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\AHE_100m\AHE_'+name.split('_')[-1]
# with rasterio.open(filename, 'w', **output_profile) as dst:
#     dst.write(resampled_data, indexes = 1)
#
# print(name)

