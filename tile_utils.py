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
import ast


def create_basic_tile_df(assets):

    df_map = assets[assets["satellite_platform"]== "s1"]
    df_map = df_map.drop_duplicates(subset=['tile_id'])
    df_map['geometry'] = df_map['geometry'].apply(ast.literal_eval)
    df_map.geometry = df_map.geometry.apply(shape)
    
    return gpd.GeoDataFrame(df_map[['tile_id', 'geometry']], crs='EPSG:4326', geometry='geometry')


def cloud_finder(tile_id, dayofyear, assets_tile_CLM, threshold = 0.2):
    
    """
    This function classifies the picture as useful = 1 or not useful = 0, regarding the rate of clouds on the band "CLM".    
    """
    
    mask = np.empty((256, 256), dtype='uint8')
    chan_rio = rasterio.open(assets_tile_CLM.query('dayofyear == @dayofyear').iloc[0,4])
    mask[:,:] = chan_rio.read()/255
    
    mask_sum = np.sum(mask)
    
    if mask_sum/(256*256) < threshold:
        return 1
    else:
        return 0
    
    
def blackout_finder(tile_id, dayofyear, assets_tile_B, threshold = 0.8):
    
    """
    This function classifies the picture as useful = 1 or not useful = 0, checking if on the band "B03" is complete.    
    """
    
    mask = np.empty((256, 256), dtype='uint8')
    chan_rio = rasterio.open(assets_tile_B.query('dayofyear == @dayofyear').iloc[0,4])
    mask[:,:] = chan_rio.read()
    
    mask_sum = np.count_nonzero(mask)
    
    if mask_sum/(256*256) > threshold:
        return 1
    else:
        return 0



def create_sunny_df(assets):

    """
    This cell takes very long. Split it if necessary.
    """
    rows = []
    
    tiles = np.sort(assets.query('asset == "CLM"')['tile_id'].unique())
    day_per_tile = dict()
    day_sunny = dict()
    s1_day_per_tile = dict()
    rate_sunny = dict()
    clean_tile = dict()
    i = 0

    for tile in tiles:
        assets_tile = assets.query(f'tile_id == {tile}')
        assets_tile_CLM = assets_tile.query(f'asset == "CLM"')
        day_per_tile[tile] = set(assets_tile_CLM['dayofyear'].unique().astype(int))
        s1_day_per_tile[tile] = set(assets.query('asset == "VH"')['dayofyear'].unique().astype(int))
        day_sunny[tile] = set()
        
        assets_tile_B = assets_tile.query(f'tile_id == {tile} & asset == "B03"') # Green channel because it is more likely to be != 0
        clean_tile[tile] = set()
        
        
        clear_output(wait=True)
        i += 1
        print(i)

        for idx, dayofyear in enumerate(day_per_tile[tile]):

            if cloud_finder(tile,dayofyear, assets_tile_CLM, 0.2) == 1:
                    day_sunny[tile].add(dayofyear) 
                    
                    if blackout_finder(tile,dayofyear, assets_tile_B, 0.2) == 1:
                        clean_tile[tile].add(dayofyear) 
                                    
        rate_sunny[tile] = len(day_sunny[tile])/len(day_per_tile[tile])
        
        rows.append([
        tile,
        s1_day_per_tile[tile],
        day_per_tile[tile],
        day_sunny[tile],
        rate_sunny[tile],
        clean_tile[tile],
        ])           

    tiles_sunny = pd.DataFrame(rows, columns = ["tile_id", "s1_days" ,"all_days", "sunny_days", "sun_rate", "clean_days"])
    
    tiles_sunny.s1_days = tiles_sunny.s1_days.apply(str)
    tiles_sunny.all_days = tiles_sunny.all_days.apply(str)
    tiles_sunny.sunny_days = tiles_sunny.sunny_days.apply(str)
    tiles_sunny.clean_days = tiles_sunny.clean_days.apply(str)
    
    return tiles_sunny


def tiles_closest(row, tiles, threshold):
    distances = tiles.distance(row.geometry.centroid).nsmallest(9)
    idxs = distances[distances < threshold].index.tolist()
    closest_ids = tiles.iloc[idxs,:].tile_id.to_list()
    try:
        closest_ids.remove(row.tile_id)
    except:
        pass
    return str(closest_ids)


def label_distribution(row, fields_train):
    tile_id = row.tile_id
    nr_total = len(fields_train.query('tile_id == @tile_id'))
    label_dist = dict()
    for label in range(1,10):
        if nr_total == 0:
            label_dist[label] = 0            
        else:
            label_dist[label] = len(fields_train.query('tile_id == @tile_id & label == @label')) / nr_total
    clear_output(wait=True)
    print(label_dist)
    return label_dist


def close_tiles_label_dist(row, tiles):
    closest_label_dist = {label:0 for label in range(1,10)}
    if len(row.tiles_closest) > 0:
        for tile_id in row.tiles_closest:
            tile_label_dist = tiles.query('tile_id == @tile_id').tile_label_dist.reset_index().iloc[0,1]
            print(tile_label_dist)
            for label in range(1, 10):
                closest_label_dist[label] = closest_label_dist[label] + tile_label_dist[str(label)]
        for label in range(1, 10):
            closest_label_dist[label] = closest_label_dist[label]/len(row.tiles_closest)
    clear_output(wait=True)
    print(closest_label_dist)
    return closest_label_dist


def read_tile_geojson(path):
    tiles = gpd.read_file(path)
    for column in tiles.select_dtypes(include='object').columns:
        try:
            tiles[column] = tiles[column].apply(eval)
        except:
            pass
    return tiles