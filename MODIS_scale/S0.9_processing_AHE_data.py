import os
import rasterio
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
import matplotlib; matplotlib.use('Qt5Agg')
# Open the GeoTIFF file

input_src = rasterio.open(r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\AHE\AHE_2010_year.tif')

filenames = os.listdir(r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\LST')
for name in filenames:
    reference_src =  rasterio.open(r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\LST\\'+name)
    reference_data = reference_src.read(1)
    transform = input_src.transform

    input_data = input_src.read(1)
    min_col, max_row, max_col, min_row = reference_src.bounds

    # Convert the spatial extent to pixel coordinates
    start_col, start_row = ~transform * (min_col, min_row)
    end_col, end_row = ~transform * (max_col, max_row)
    start_col, start_row, end_col, end_row = map(round, (start_col, start_row, end_col, end_row))

    # Extract the corresponding portion of the input raster
    extracted_data = input_data[start_row:end_row, start_col:end_col]

    scale_factor_y = (reference_data.shape[0])/extracted_data.shape[0]
    scale_factor_x = (reference_data.shape[1])/extracted_data.shape[1]

    # Resample the source raster using bilinear interpolation
    resampled_data = zoom(extracted_data, (scale_factor_y, scale_factor_x), order=1)

    # Update the transform matrix to reflect the new bounds
    transform = reference_src.transform

    # Create a new raster file for the extracted data
    output_profile = reference_src.profile
    output_profile.update(count=1, dtype='float32')
    filename = r'D:\Projects\project 1 urban vegetation thermoregulation\1_Input\AHE_100m\AHE_'+name.split('_')[-1]
    with rasterio.open(filename, 'w', **output_profile) as dst:
        dst.write(resampled_data, indexes = 1)

    print(name)

