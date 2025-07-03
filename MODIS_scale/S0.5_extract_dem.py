import sys
sys.path.insert(0, r'C:\Users\zhengfeiguo\Desktop\Current Projects\codes_python\\')
from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize()

# Part 1: Main function

water_mask = ee.Image(1).subtract(ee.Image('MERIT/Hydro/v1_0_1').select('wat'))

# S1. unmixing greenspace fraction

FVC = ee.Image('WWF/HydroSHEDS/15CONDEM').select('b1')

dfc = gpd.read_file('D:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/urban_100km2.shp')
# dfc = gpd.read_file('C:/Projects/Postdoc urban greening/Data/urban_cores_newtowns/GUB_urban_cores.shp')
dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']

folder = 'DEM_100m'
for city in cities:#1087

    geo = dfc[dfc['ID'] == city].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'dem_'+str(geo['ID'][0])
    print(dscr)
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                     region=geometry,  # an ee.Geometry object.
                                     description=dscr,
                                     folder=folder,
                                     fileNamePrefix=dscr,
                                     scale = 100,
                                     crs = 'epsg:4326')
    task.start()


# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())

