"""
functions for creating a csv-file where all paths to labels, field_ids, images, etc. is stored
"""

import rasterio
import json
from IPython.display import clear_output
import pandas as pd
import geopandas as gpd

from shapely.geometry import shape
from shapely.geometry import Polygon

from osgeo import gdal
import elevation
import numpy as np
from rasterio.mask import mask

def create_tile_stacked(tile_id, dayofyear, path_write, assets_tile_df, asset_list):
    tile_filepath = list(assets_tile_df.iloc[:,4])
    
    # Read metadata of first file
    with rasterio.open(tile_filepath[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(tile_filepath))

    # Read each layer and write it to stack
    with rasterio.open(path_write, 'w', **meta) as dst:
        for id, layer in enumerate(tile_filepath, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))

                
def create_stacked(assets_df ,tiles, satellite,num_days):
    tile_ids = assets_df['tile_id'].unique()
    
    asset_list = assets_df.query(f'satellite_platform == {satellite}')['asset'].unique()
    assets_df_temp = pd.DataFrame(columns=['tile_id','datetime','satellite_platform','asset','file_path','date','month','dayofyear'])

    for tile_id in tile_ids:
        clear_output(wait=True)
        print(tile_id)
        
        if satellite == "s2":
            days_available = "clean_days"
            
        elif satellite == "s1":
            days_available = "s1_days"
            
        clean_days = sorted(list(tiles.query('tile_id == @tile_id')[days_available].iloc[0]))
        assist_select = np.linspace(0, len(clean_days)-1, num = num_days, dtype=np.int)
        clean_days_select = np.array(clean_days)[assist_select]

        for day in clean_days_select:
            assets_tile_df = assets_df.query(f'tile_id == @tile_id & dayofyear == @day & satellite_platform == {satellite} & asset in @asset_list')
            # create path to save file to
            #path_write = "/".join(assets_tile_df.iloc[0,4].split("/")[0:-1]) + '/s2_stacked.tif'
            
            path_write = f'/stacked_files/{satellite}/{tile_id}_{satellite}_{day}_stacked.tif'
            
            create_tile_stacked(tile_id, day, path_write, assets_tile_df, asset_list)

            row_append = dict(assets_tile_df.iloc[0])
            row_append['asset'] = f'{satellite}_stacked'
            row_append['file_path'] = path_write
            assets_df_temp = assets_df_temp.append(row_append, ignore_index=True)
        
    return assets_df_temp