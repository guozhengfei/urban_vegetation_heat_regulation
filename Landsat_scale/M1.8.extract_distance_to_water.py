import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
import glob
import numpy as np
# Paths
current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')

dtw_path = current_dir+'/2_Output/distance2water_30arcsec.tif'  # Update with your global dtw file
lst_folder = current_dir+'/1_Input/LST/'  # Update with your LST folder
output_folder = current_dir+'/1_Input/Distance_to_water/'  # Update with your output folder

os.makedirs(output_folder, exist_ok=True)

# List all LST city GeoTIFFs
lst_files = glob.glob(os.path.join(lst_folder, '*.tif'))

for lst_file in lst_files:
    with rasterio.open(lst_file) as src_lst:
        lst_meta = src_lst.meta.copy()
        lst_crs = src_lst.crs
        lst_transform = src_lst.transform
        lst_width = src_lst.width
        lst_height = src_lst.height

        # Open global dtw
        with rasterio.open(dtw_path) as src_dtw:
            # Prepare output array
            dtw_city = np.empty((lst_height, lst_width), dtype=src_dtw.dtypes[0])

            # Reproject dtw to match LST
            reproject(
                source=rasterio.band(src_dtw, 1),
                destination=dtw_city,
                src_transform=src_dtw.transform,
                src_crs=src_dtw.crs,
                dst_transform=lst_transform,
                dst_crs=lst_crs,
                resampling=Resampling.nearest
            )

        # Save city-scale dtw
        dtw_city_path = os.path.join(
            output_folder,
            os.path.basename(lst_file).replace('Landsat', 'dtw')
        )
        lst_meta.update(dtype=dtw_city.dtype, count=1)
        with rasterio.open(dtw_city_path, 'w', **lst_meta) as dst:
            dst.write(dtw_city, 1)

print("City-scale DTW extraction complete.")