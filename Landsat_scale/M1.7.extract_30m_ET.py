
from geeCodes import *

import ee
#earthengine authenticate
ee.Authenticate()
ee.Initialize(project = 'ee-zhengfei')

current_dir = os.path.dirname(os.path.dirname(os.getcwd())).replace('\\', '/')

dataset = ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0') \
  .filterDate('2020-01-01', '2021-01-01')
FVC = dataset.select('et_ensemble_mad').sum().toFloat()

dfc = gpd.read_file(current_dir + '/1_Input/urban_cores_newtowns/urban_100km2.shp')


dfc = dfc.to_crs("EPSG:4326")
cities = dfc['ID']
for id in [1027,858,807,1080,867,936,992,785]: #767
    geo = dfc[dfc['ID'] == id].reset_index(drop=True)
    S = returnCityBoundary(geo)

    ####### GEE Geometry ###############################################
    geometry = returnGeometry(S[0])
    fvc = FVC.clip(geometry);

    dscr = 'ET_2020_30m_'+str(geo['ID'][0])
    print(dscr)
    folder = 'export_ET'
    task = ee.batch.Export.image.toDrive(image=fvc,  # an ee.Image object.
                                        region=geometry,  # an ee.Geometry object.
                                        description=dscr,
                                        folder=folder,
                                        fileNamePrefix=dscr,
                                        scale = 30,
                                        crs = 'epsg:4326')
    task.start()


# linearFit = FVC.select(['system:time_start', 'veg_frac']).reduce(ee.Reducer.linearFit())

