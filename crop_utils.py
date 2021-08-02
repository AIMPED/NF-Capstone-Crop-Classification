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

channel_to_number = {"B01":0, "B02":1, "B03":2, "B04":3, "B05":4, "B06":5, "B07":6, "B08":7,
                     "B09":8, "B11":9, "B12":10, "B8A":11, "CLM":None, "BS_IDX": 12, "MOIST_IDX": 13, "NDVI_IDX": 14}
number_to_channel = {number: channel for channel, number in channel_to_number.items()}


def create_ndviindex(image, channel_to_number):
    image_B08 = image[channel_to_number['B08'],:,:]
    image_B04 = image[channel_to_number['B04'],:,:]
    non_zero = (image_B08 + image_B04 != 0)
    ndvi_index = np.zeros((image.shape[1], image.shape[2]))
    ndvi_index[non_zero] = np.nan_to_num((image_B08[non_zero] - image_B04[non_zero]) / (image_B08[non_zero] + image_B04[non_zero]))
    
    upper_limit = 1
    lower_limit = -0.2
    
    ndvi_index = np.where(ndvi_index > lower_limit, ndvi_index , lower_limit)
    ndvi_index = np.where(ndvi_index < upper_limit, ndvi_index , upper_limit)
    
    return ndvi_index

def create_moistureindex(image, channel_to_number):
    image_B8A = image[channel_to_number['B8A'],:,:]
    image_B11 = image[channel_to_number['B11'],:,:]
    non_zero = (image_B8A + image_B11 != 0)
    moisture_index = np.zeros((image.shape[1], image.shape[2]))
    moisture_index[non_zero] = np.nan_to_num((image_B8A[non_zero] - image_B11[non_zero]) / (image_B8A[non_zero] + image_B11[non_zero]))
    
    upper_limit = 0.8
    lower_limit = -0.8
    
    moisture_index = np.where(moisture_index > lower_limit, moisture_index , lower_limit)
    moisture_index = np.where(moisture_index < upper_limit, moisture_index , upper_limit)
    
    return moisture_index

def create_baresoilindex(image, channel_to_number):
    
    # Other option found: NBSI = ((B11 + B04)-(B08 + B02))/((B11 + B04)+(B08 + B02))
    
    image_B08 = image[channel_to_number['B08'],:,:]
    image_B03 = image[channel_to_number['B03'],:,:]
    image_B04 = image[channel_to_number['B04'],:,:]
    
    non_zero = (image_B08 - image_B03 - image_B04 != 0)
    baresoil_index = np.zeros((image.shape[1], image.shape[2]))
    baresoil_index[non_zero] = np.nan_to_num((image_B08[non_zero] + image_B03[non_zero] + image_B04[non_zero]) / (image_B08[non_zero] - image_B03[non_zero] - image_B03[non_zero]))
    
    return baresoil_index

def mean_var(image, number_to_channel):
    """
    extracts mean and variance for each layer of 'image'
    
    Arguments: - 'src': np.array in the form of (nr_channels, height_image, width_image)
    """
    
    # count nr. of channels in 'src'
    nr_chan = image.shape[0]
    
    mean = {number_to_channel[chan] + "_MEAN": 0 for chan in range(0, nr_chan)}
    var = {number_to_channel[chan] + "_VAR": 0 for chan in range(0, nr_chan)}
    
    for chan in range(0, nr_chan):
        data = image[chan, image[chan, :, :] != 0]
        if(len(data.flatten()) > 0):
            mean[number_to_channel[chan] + "_MEAN"] = np.mean(data.flatten())
            var[number_to_channel[chan] + "_VAR"] = np.var(data.flatten())
    
    return (mean, var)


import rasterio.features
from rasterio.plot import show
from rasterio.mask import mask

def mask_mean_var(image_stacked, field_ids_tile, field_id, channel_to_number, number_to_channel):
    """
    extracts mean and variance for each layer of 'path' after masking it with the geometry of 'geom'
    

    """
    image_stacked = np.delete(image_stacked, 12, axis=0)
    
    mask = (field_ids_tile != field_id)
    
    top = np.min(np.where(np.sum(mask, axis=1) != mask.shape[1]))
    bottom = np.max(np.where(np.sum(mask, axis=1) != mask.shape[1]))
    left = np.min(np.where(np.sum(mask, axis=0) != mask.shape[0]))
    right = np.max(np.where(np.sum(mask, axis=0) != mask.shape[0]))
    
    mask_broad = np.broadcast_to(mask, image_stacked.shape)
    
    image_masked = ma.masked_array(image_stacked, mask=mask_broad).filled(fill_value=0)
    image_cropped = image_masked[:,top:(bottom+1),left:(right+1)]
    
    # remove cloud coverage channel    
      
    image_with_idx = np.zeros((image_cropped.shape[0] + 3, image_cropped.shape[1], image_cropped.shape[2]))
    image_with_idx[0:image_cropped.shape[0],:,:] = image_cropped
    
    image_with_idx[channel_to_number['BS_IDX'],:,:] = create_baresoilindex(image_cropped, channel_to_number)
    image_with_idx[channel_to_number['MOIST_IDX'],:,:] = create_moistureindex(image_cropped, channel_to_number)
    image_with_idx[channel_to_number['NDVI_IDX'],:,:] = create_ndviindex(image_cropped, channel_to_number)
    
    mean, var = mean_var(image_with_idx, number_to_channel)
    return mean, var    


"""
crop the images with a different technique, get the mask from the label.tif file directly
"""
import numpy.ma as ma

def mask_mean_var_wrapper(row, image_stacked, field_ids_tile, channel_to_number, number_to_channel, day):
    mean, var = mask_mean_var(image_stacked, field_ids_tile[0,:,:], row.field_id, channel_to_number, number_to_channel)
    mean_var_day = {key + "_" + str(day): value for key, value in {**mean, **var}.items()}
    return mean_var_day

def get_sun_labeldist(tile_id, nr_fields, tiles_train):
    tile = tiles_train.query('tile_id == ' + str(tile_id)).iloc[0]
    dict_temp = {'neighboor_label_' + key: value for key, value in tile.close_label_dist.items()}
    dict_temp['sun_rate'] = tile.sun_rate
    
    df_temp = pd.DataFrame(dict_temp, index=[0])
    
    return pd.DataFrame(np.repeat(df_temp.values, nr_fields, axis=0), columns=df_temp.columns)


def create_mean_var(assets_df, assets_stacked_df, fields, tiles):
    tile_ids = assets_df['tile_id'].unique()
    fields_mean_var = pd.DataFrame()

    i = 0
    for tile_id in tile_ids:
        i += 1
        clear_output(wait=True)
        print(f'Tile Nr. {i} of {len(tile_ids)}')

        tile_df = assets_stacked_df.query('tile_id == @tile_id')
        tile_days = list(tile_df['dayofyear'])

        tile_fields_df = fields.query('tile_id == @tile_id')
        right_df = get_sun_labeldist(tile_id, tile_fields_df.shape[0], tiles)
        right_df.index = tile_fields_df.index
        tile_fields_df = pd.concat([tile_fields_df, right_df], axis=1) 

        field_ids_tile = rasterio.open(assets_df.query('tile_id == @tile_id & asset == "field_ids"').iloc[0,4]).read()

        for day_nr, day  in enumerate(tile_days):
            path = tile_df.query('dayofyear == @day').iloc[0,4]
            
            image_stacked = rasterio.open(path).read()
            right_df = tile_fields_df.apply(mask_mean_var_wrapper, args=(image_stacked, field_ids_tile, channel_to_number, number_to_channel,
                                                                          day_nr,),axis='columns', result_type='expand')
            tile_fields_df = pd.concat([tile_fields_df, right_df], axis=1)   
            
            
        fields_mean_var = pd.concat([fields_mean_var, tile_fields_df], axis = 0)
        clear_output(wait=True)

    return fields_mean_var