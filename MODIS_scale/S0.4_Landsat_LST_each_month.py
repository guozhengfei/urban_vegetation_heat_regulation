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
dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
projection_Landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate('2015-01-01', '2016-01-01')\
                .first().projection()
projection_modis = ee.ImageCollection("MODIS/061/MOD13A3").filterDate('2015-01-01','2016-01-01').first().projection().atScale(100)
folder = 'landsat_lst'
cities=[88,89,90,165,167,168,442,443,444,583,584,586,588,589,590,591,592,593,646,647,649,699,700,701,702,703,706,707,714,715,716,719,721,722,723,724,725,726,727]
for city in cities:#1087  1126

    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    # geometry = dfc[dfc['ID'] == city]['geometry'][0]
    S = returnCityBoundary(geo)
    geometry = returnGeometry(S[0])

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
    reduced_image = month_12.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=12000,
        bestEffort=True
    ).reproject(
        crs=projection_modis,
        scale=100
    )

    # .reproject('EPSG:4326', None, 100)

    dscr = ('Landsat_') + str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=reduced_image.clip(geometry),  # an ee.Image object.
                                         region=geometry,  # an ee.Geometry object.
                                         description=dscr,
                                         folder=folder,
                                         fileNamePrefix=dscr,
                                         scale=100,
                                         crs='epsg:4326')
    task.start()


