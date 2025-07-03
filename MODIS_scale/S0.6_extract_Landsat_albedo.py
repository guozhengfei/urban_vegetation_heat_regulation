import sys
sys.path.insert(0, r'D:\Current Projects\codes_python\\')
from geeCodes import *

import ee
import geopandas as gpd
# Initialize Earth Engine
ee.Initialize()

projection_modis = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection().atScale(100)

# Define the maskL8sr function
def maskL8sr(image):
    # Bit 0 - Fill
    # Bit 1 - Dilated Cloud
    # Bit 2 - Cirrus
    # Bit 3 - Cloud
    # Bit 4 - Cloud Shadow
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select('QA_RADSAT').eq(0)

    # Apply the scaling factors to the appropriate bands.
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0).subtract(273.15)

    # Replace the original bands with the scaled ones and apply the masks.
    return image.addBands(opticalBands, None, True) \
        .addBands(thermalBands, None, True) \
        .updateMask(qaMask) \
        .updateMask(saturationMask)

# Map the function over one year of data.
image = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterDate('2018-01-01', '2022-01-01') \
    .map(maskL8sr) \
    .median()

albedo = image.expression(
    '(0.356*blue + 0.13*red + 0.373*nir + 0.085*sw1 + 0.072*sw2 - 0.0018) / 1.016',
    {
        'nir': image.select('SR_B5'),  # Near-Infrared band
        'red': image.select('SR_B4'),  # Red band
        'blue': image.select('SR_B2'),  # Blue band
        'sw1': image.select('SR_B6'),
        'sw2': image.select('SR_B7'),
    }
)

dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'albedo_Landsat_100m'
for city in cities:#1087

    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = albedo.clip(geometry).setDefaultProjection(projection_modis);
    reduced_image = fvc.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=12000,
        bestEffort=True
    ).reproject(
        crs=projection_modis,
        scale=100
    )

    dscr = 'albedo_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=reduced_image,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale = 100,
                                     crs = 'epsg:4326')
    task.start()


# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())