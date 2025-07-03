import sys
sys.path.insert(0, r'D:\Current Projects\urban_cooling_efficiency_one_year\\')
from geeCodes import *
import ee
import geopandas as gpd
# Initialize Earth Engine
ee.Initialize()

# Define the function to mask Landsat 8 Surface Reflectance

# Map over years to calculate composite for each year
dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
projection_Landsat = ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDVI') \
                .filterDate('2017-01-01', '2018-01-01')\
                .first().projection()
projection_modis = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection().atScale(100)
folder = 'landsat_ndvi'
# cities=[88,89,90,165,167,168,442,443,444,583,584,586,588,589,590,591,592,593,646,647,649,699,700,701,702,703,706,707,714,715,716,719,721,722,723,724,725,726,727]
for city in cities:#1087  1126

    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)
    geometry = returnGeometry(S[0])

    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDVI') \
        .filterDate('2017-01-01', '2021-01-01') \
        .mean()

    composite = collection.select(['NDVI'])

    month_12 = composite.clip(geometry).setDefaultProjection(projection_modis)

    reduced_image = month_12.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=12000,
        bestEffort=True
    ).reproject(
        crs=projection_modis,
        scale=100
    )

    dscr = ('Landsat_ndvi_') + str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=reduced_image.clip(geometry),  # an ee.Image object.
                                         region=geometry,  # an ee.Geometry object.
                                         description=dscr,
                                         folder=folder,
                                         fileNamePrefix=dscr,
                                         scale=100,
                                         crs='epsg:4326')
    task.start()


