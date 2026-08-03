"""
scripts/gee_export_spi.py — Export CHIRPS monthly rainfall per India district from GEE.
 
SPI-3 (3-month Standardised Precipitation Index) is computed from this data
in Step 33 locally using the downloaded CSV.
 
Usage: python3 scripts/gee_export_spi.py
Output: data/raw/external/chirps_monthly_districts.csv
"""
 
import ee
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_console_logger
 
log = get_console_logger(__name__)
 
def main():
    ee.Initialize(project='malnutrisense')
 
    india = ee.FeatureCollection('FAO/GAUL/2015/level2').filter(
        ee.Filter.eq('ADM0_NAME', 'India')
    )
 
    # CHIRPS daily rainfall — aggregate to monthly total
    chirps_monthly = (
        ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
        .filterDate('2018-01-01', '2021-12-31')  # extra year for SPI baseline
        .filterBounds(india.geometry())
    )
 
    # Compute monthly totals (sum of daily values per month)
    def monthly_sum(year, month):
        start = ee.Date.fromYMD(year, month, 1)
        end   = start.advance(1, 'month')
        return (
            chirps_monthly.filterDate(start, end)
            .sum()
            .set('year', year)
            .set('month', month)
        )
 
    # Build collection of monthly images
    months = ee.List.sequence(1, 12)
    years  = ee.List.sequence(2018, 2021)
 
    monthly_images = ee.ImageCollection(
        years.map(lambda y: months.map(lambda m: monthly_sum(y, m))).flatten()
    )
 
    # Reduce to district mean precipitation per month
    district_rain = monthly_images.map(lambda img:
        img.reduceRegions(
            collection=india,
            reducer=ee.Reducer.mean(),
            scale=5566,  # CHIRPS resolution ~5km
        ).map(lambda f: f.set('year', img.get('year'))
               .set('month', img.get('month')))
    ).flatten()
 
    task = ee.batch.Export.table.toDrive(
        collection=district_rain,
        description='india_chirps_monthly_2018_2021',
        folder='malnutrisense_gee',
        fileNamePrefix='chirps_monthly_districts',
        fileFormat='CSV',
        selectors=['ADM1_NAME','ADM2_NAME','year','month','mean'],
    )
    task.start()
    print('CHIRPS export started — check GEE Tasks tab')
    print('Download to: data/raw/external/chirps_monthly_districts.csv')
 
if __name__ == '__main__':
    main()
