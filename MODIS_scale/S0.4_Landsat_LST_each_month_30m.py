import sys
sys.path.insert(0, r'D:\Current Projects\urban_cooling_efficiency_one_year\\')
from geeCodes import *
import ee
import geopandas as gpd
# Initialize Earth Engine
ee.Initialize()

# Define the function to mask Landsat 8 Surface Reflectance
def maskL8sr(image):
    qaMask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select('QA_RADSAT').eq(0)

    # Apply the scaling factors to the appropriate bands
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0).subtract(273.15)

    # Replace the original bands with the scaled ones and apply the masks
    return image.addBands(opticalBands, None, True) \
        .addBands(thermalBands, None, True) \
        .updateMask(qaMask) \
        .updateMask(saturationMask)

# Map over years to calculate composite for each year
projection_Landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate('2015-01-01', '2016-01-01')\
                .first().projection()
projection_modis = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection()
folder = 'landsat_lst_30m'

# geometry = ee.Geometry.Polygon(
#     [[[20.04748964697596, 32.0757081981845],
#       [20.04748964697596, 32.06781682747744],
#       [20.05675936133143, 32.06781682747744],
#       [20.05675936133143, 32.0757081981845]]]); # warmer grass

geometry = ee.Geometry.Polygon([[[20.201937783797984, 32.0961881688341],
          [20.201937783797984, 32.09153447641958],
          [20.208332170089488, 32.09153447641958],
          [20.208332170089488, 32.0961881688341]]]); # warmer cropland

geometry = ee.Geometry.Polygon([[[46.3342712915658, 38.06441004418234],
           [46.3342712915658, 38.06016952361518],
           [46.34053693182459, 38.06016952361518],
           [46.34053693182459, 38.06441004418234]]]); # warmer tree

geometry = ee.Geometry.Polygon([[[121.53434724580026, 25.029807803106696],
           [121.53434724580026, 25.02234164258371],
           [121.54297322999216, 25.02234164258371],
           [121.54297322999216, 25.029807803106696]]]); # cooler tree

geometry = ee.Geometry.Polygon([[[121.04792774373527, 14.31241960453926],
           [121.04792774373527, 14.301191820799517],
           [121.05985820943351, 14.301191820799517],
           [121.05985820943351, 14.31241960453926]]]); # cooler crop

geometry = ee.Geometry.Polygon([[[2.490425428781289, 48.826865828351615],
           [2.490425428781289, 48.82028250729713],
           [2.501454672250527, 48.82028250729713],
           [2.501454672250527, 48.826865828351615]]]); # cooler grass

month_12 = []
for month in range(12):
    month_coll = []
    for year in range(2015,2022):
        startYear = ee.Date.fromYMD(year, month+1, 1)
        endYear = startYear.advance(1, 'month')

        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterBounds(geometry) \
            .filterDate(startYear, endYear) \
            .map(maskL8sr) \
            .select('ST_B10')
        composite = collection.median()
        month_coll.append(composite)

    month_image = ee.ImageCollection.fromImages(month_coll).median()
    band_names = month_image.bandNames().getInfo()
    if band_names != ['ST_B10']: continue

    month_12.append(month_image)

month_12 = ee.ImageCollection.fromImages(month_12).select('ST_B10').toBands().clip(geometry).setDefaultProjection(projection_modis)
# property_names = month_12.propertyNames().getInfo()
# print(property_names)
reduced_image = month_12
# .reproject('EPSG:4326', None, 100)

# dscr = 'Landsat30m_grass_warmer'
# dscr = 'Landsat30m_cropland_warmer'
# dscr = 'Landsat30m_tree_warmer'
# dscr = 'Landsat30m_tree_cooler'
# dscr = 'Landsat30m_cropland_cooler'
dscr = 'Landsat30m_grass_cooler'

print(dscr)
task = ee.batch.Export.image.toDrive(image=reduced_image.clip(geometry),  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale=30,
                                     crs='epsg:4326')
task.start()


