import ee
import geemap
import pandas as pd
from datetime import datetime, timedelta

def initialize_earth_engine():
    """Authenticate and initialize Google Earth Engine"""
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()

def get_sentinel_data(geometry, start_date, end_date, cloud_threshold=20):
    """Collect Sentinel-2 imagery for a given region and time period"""
    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(geometry)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
    
    return collection

def create_time_series(region, years=5):
    """Generate monthly composites over multiple years"""
    time_series = []
    
    for year in range(datetime.now().year - years, datetime.now().year):
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = (datetime(year, month, 1) + timedelta(days=31)).strftime("%Y-%m-%d")
            
            monthly_composite = (get_sentinel_data(region, start, end)
                                .median()
                                .clip(region))
            
            time_series.append(monthly_composite.set('year', year).set('month', month))
