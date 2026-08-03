"""
scripts/gee_export_ndvi.py — Export mean NDVI per India district from GEE.
 
Prerequisites:
  pip install earthengine-api
  earthengine authenticate --auth_mode=notebook (run once, opens browser for auth)
 
Usage: python3 scripts/gee_export_ndvi.py
Output: Exports CSV to Google Drive → download to data/raw/external/ndvi_districts.csv
"""
 
import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
def main():
    # Authenticate and initialize GEE
    ee.Authenticate()   # opens browser first time only
    ee.Initialize(project='malnutrisense')  # replace with your project ID
    log.info('GEE initialized')
 
    # Load India district shapefile uploaded to GEE as asset
    # OR use the built-in FAO GAUL boundaries
    india = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(
        ee.Filter.eq('ADM0_NAME', 'India')
    )
    log.info(f'India districts loaded from GAUL')
 
    # Load MODIS MOD13Q1 NDVI — 16-day composite, 250m resolution
    ndvi_collection = (
        ee.ImageCollection('MODIS/061/MOD13Q1')
        .filterDate('2019-01-01', '2021-12-31')  # NFHS-5 survey period
        .filterBounds(india.geometry())
        .select('NDVI')                          # NDVI band
        .map(lambda img: img.multiply(0.0001))  # scale factor: raw × 0.0001
    )
 
    # Compute 2-year mean NDVI per pixel
    mean_ndvi = ndvi_collection.mean()
 
    # Reduce to district-level mean NDVI
    district_ndvi = mean_ndvi.reduceRegions(
        collection=india,
        reducer=ee.Reducer.mean(),
        scale=250,   # 250m resolution
    )
 
    # Export to Google Drive as CSV
    task = ee.batch.Export.table.toDrive(
        collection=district_ndvi,
        description='india_district_ndvi_2019_2021',
        folder='malnutrisense_gee',
        fileNamePrefix='ndvi_districts',
        fileFormat='CSV',
        selectors=['ADM1_NAME','ADM2_NAME','mean'],  # state, district, NDVI
    )
    task.start()
    log.info('NDVI export task started. Check GEE Tasks tab for progress.')
    log.info('Download CSV from Google Drive when complete.')
    log.info('Save to: data/raw/external/ndvi_districts.csv')
    print('Export task started — check https://code.earthengine.google.com/tasks')
 
if __name__ == '__main__':
    main()
