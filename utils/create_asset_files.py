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

import os





def resolve_path(base, path):
    return Path(os.path.join(base, path)).resolve()


def create_assets(collection_id):

    collection = json.load(open(f'{collection_id}/collection.json', 'r'))
    rows = []
    item_links = {}
    
    counter = 0
    for i, link in enumerate(collection['links']):
        
        if link['rel'] != 'item':
            continue
        item_links[i] = link['href']
        
    for index, item_link in item_links.items():
        counter += 1
        clear_output(wait=True)
        print(f"Procesing {counter} of {len(item_links)}")
        item_path = f'{collection_id}/{item_link}'
        current_path = os.path.dirname(item_path)
        item = json.load(open(item_path, 'r'))
        tile_id = item['id'].split('_')[-1]
        for asset_key, asset in item['assets'].items():
            rows.append([
                tile_id,
                None,
                None,
                asset_key,
                str(resolve_path(current_path, asset['href'])),
                None
            ])
            
        for link in item['links']:
            if link['rel'] != 'source':
                continue
            link_path = resolve_path(current_path, link['href'])
            source_path = os.path.dirname(link_path)
            try:
                source_item = json.load(open(link_path, 'r'))
            except FileNotFoundError:
                continue
            datetime = source_item['properties']['datetime']
            satellite_platform = source_item['collection'].split('_')[-1]
            geometry = source_item["geometry"]
            for asset_key, asset in source_item['assets'].items():
                rows.append([
                    tile_id,
                    datetime,
                    satellite_platform,
                    asset_key,
                    str(resolve_path(source_path, asset['href'])),
                    geometry
                ])
    return pd.DataFrame(rows, columns=['tile_id', 'datetime', 'satellite_platform', 'asset', 'file_path', "geometry"])





def create_field_list_train(assets_df):

    tile_id_list = assets_df['tile_id'].unique()

    fields_list = []

    i = 0
    for tile_id in tile_id_list:

        labels_rio = rasterio.open(assets_df.query('tile_id == @tile_id & asset == "labels"').iloc[0,4])

        field_ids_rio = rasterio.open(assets_df.query('tile_id == @tile_id & asset == "field_ids"').iloc[0,4])

        labels = labels_rio.read()
        field_ids = field_ids_rio.read().astype('float32')

        # Read the dataset's valid data mask as a ndarray.
        mask_labels = (labels != 0)

        i += 1   
        clear_output(wait=True)
        print(f"Tile Nr. {i} of total " + str(len(tile_id_list)))


        for geom, val in rasterio.features.shapes(field_ids, mask_labels, transform=field_ids_rio.transform):

            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            geom = rasterio.warp.transform_geom(
                field_ids_rio.crs, 'EPSG:4326', geom, precision=6)

            fields_list.append([shape(geom), int(val), round(np.mean(labels[field_ids == val])), tile_id])

            # Print GeoJSON shapes to stdout.
        
    fields_train = pd.DataFrame(fields_list)
    fields_train.columns = ['geometry', 'field_id', 'label', 'tile_id']

    fields_train = gpd.GeoDataFrame(fields_train, crs='EPSG:4326', geometry='geometry')
        
    fields_crs32634 = fields_train.to_crs(32634)
    fields_train['field_area_km2'] = fields_crs32634.area/1000000
        
    return fields_train



def create_field_list_test(assets_df):
    
    tile_id_list = assets_df['tile_id'].unique()

    fields_list = []

    i = 0
    for tile_id in tile_id_list:

        field_ids_rio = rasterio.open(assets_df.query('tile_id == @tile_id & asset == "field_ids"').iloc[0,4])

        field_ids = field_ids_rio.read().astype('float32')

        # Read the dataset's valid data mask as a ndarray.
        mask_labels = (field_ids != 0)

        i += 1   
        clear_output(wait=True)
        print(f"Tile Nr. {i} of total " + str(len(tile_id_list)))


        for geom, val in rasterio.features.shapes(field_ids, mask_labels, transform=field_ids_rio.transform):

            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            geom = rasterio.warp.transform_geom(
                field_ids_rio.crs, 'EPSG:4326', geom, precision=6)

            fields_list.append([shape(geom), int(val), tile_id])

            # Print GeoJSON shapes to stdout.
        
        
    fields_test = pd.DataFrame(fields_list)
    fields_test.columns = ['geometry', 'field_id', 'tile_id']

    fields_test = gpd.GeoDataFrame(fields_test, crs='EPSG:4326', geometry='geometry')
    
    fields_crs32634 = fields_test.to_crs(32634)
    fields_test['field_area_km2'] = fields_crs32634.area/1000000
        
        
    return fields_test



def download_elevation(fields_train):
    bound_minx = np.min(fields_train.bounds.minx) - 0.2
    bound_miny = np.min(fields_train.bounds.miny) - 0.2
    bound_maxx = np.max(fields_train.bounds.maxx) + 0.2
    bound_maxy = np.max(fields_train.bounds.maxy) + 0.2
    elevation.clip(bounds=(bound_minx, bound_miny, bound_maxx,  bound_maxy-2 ), output='/home/jupyter/NF-Capstone-Crop-Classification/data/elev_south.tif')    
    elevation.clip(bounds=(bound_minx, bound_maxy-2 , bound_maxx,  bound_maxy ), output='/home/jupyter/NF-Capstone-Crop-Classification/data/elev_north.tif')
    
    vrt = gdal.BuildVRT('/home/jupyter/NF-Capstone-Crop-Classification/merged.vrt', ['data/elev_north.tif', 'data/elev_south.tif'])
    gdal.Translate("/home/jupyter/NF-Capstone-Crop-Classification/elev_merged.tif", vrt)
    
    
    

def get_elev(shape, elev_src, elev_read):
    out_image, _ = mask(elev_src, shapes=[shape], crop=True)
    ret = np.mean(out_image[out_image != -32768])
    if np.isnan(ret) == False:
        return ret
    else:
        coords = shape.centroid.coords[0]
        index = elev_src.index(coords[0],coords[1])
        return elev_read[0,index[0],index[1]]
    

def remove_dupl_fields(fields):
    label0_idxs = fields.query('field_id == 0').index
    fields = fields.drop(label0_idxs, axis=0)
    
    duplicates = fields[fields.field_id.duplicated(keep=False)]
    
    fields.drop_duplicates(subset='field_id', keep=False, inplace=True)
    
    for field_id in duplicates.field_id.unique():
        id_duplicated = duplicates.query('field_id == @field_id')
        keep = id_duplicated.sort_values('field_area_km2', ascending=False).iloc[0,:]
        fields = fields.append(keep)
    
    return fields