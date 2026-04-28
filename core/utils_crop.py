##################################################
##################################################
import os
import gc
import csv
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from dask.diagnostics import ProgressBar

from core.paths import *

##################################################
##  DATA HANDLING
##################################################

## https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
## reference: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
def calc_earth_cellarea(lat_mid, dlat, dlon, debug=False):
    '''
    Function to calculate grid cell area of earth surface
    Input:
    ------
    lat_mid (np.array | float)      latitude of grid cell
    dlat (float)                    latitude interval
    dlon (float)                    longitude interval
    
    Output:
    ------
    area (np.array | float)         surface area of earth
    
    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    print(rf'Caution: unit of calculated area is km²!')

    # equatorial radius (km)
    a_Earth = 6378.137
    # polar radius (km)
    b_Earth = 6356.752
    # square of first eccentricity
    e2 = (a_Earth**2-b_Earth**2)/a_Earth**2
    # grid cell area (km2)
    area = a_Earth**2*(1-e2)/(1-e2*np.sin(np.radians(lat_mid))**2)**2*np.radians(dlat)*np.radians(dlon)*np.cos(np.radians(lat_mid))
    return area

def calc_bic(y, y_mod, p, debug=False):
    '''
    Fucntion to calculate the Bayesian information criterion
    
    Input:
    ------
    y (np.ndarray | xr.DataArray)               observed data
    y_mod (np.ndarray | xr.DataArray)           fitted data
    p (int)                                     number of parameters

    Output:
    -------
    BIC (float)                                 Bayesian information criterion

    Options:
    --------
    debug (bool)                                whether or not to print debug information
                                                default = False

    '''
    assert len(y) == len(y_mod)
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    if isinstance(y, xr.DataArray): y = y.values.flatten()
    elif isinstance(y, np.array): y = y.flatten()
    if isinstance(y_mod, xr.DataArray): y_mod = y_mod.values.flatten()
    elif isinstance(y_mod, np.array): y_mod = y_mod.flatten()
    y[np.isnan(y_mod)] = np.nan
    y_mod[np.isnan(y)] = np.nan
    y = y[~np.isnan(y)]
    y_mod = y_mod[~np.isnan(y_mod)]
    n = len(y)
    if n == 0: 
        return np.nan, n
    else:
        SSE = np.sum(((y - y_mod)**2))
        BIC = n*np.log(SSE/n) + p*np.log(n)
        return BIC, n

def calc_r2(y, y_mod, debug=False):
    '''
    Fucntion to calculate the Bayesian information criterion
    
    Input:
    ------
    y (np.ndarray | xr.DataArray)               observed data
    y_mod (np.ndarray | xr.DataArray)           fitted data

    Output:
    -------
    R2 (float)                                  R squared

    Options:
    --------
    debug (bool)                               whether or not to print debug information
                                                default = False

    '''   
    assert len(y) == len(y_mod)
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    if isinstance(y, xr.DataArray): y = y.values.flatten()
    elif isinstance(y, np.ndarray): y = y.values.flatten()
    if isinstance(y_mod, xr.DataArray): y_mod = y_mod.values.flatten()
    elif isinstance(y_mod, np.ndarray): y_mod = y_mod.values.flatten()
    y[np.isnan(y_mod)] = np.nan
    y_mod[np.isnan(y)] = np.nan
    y = y[~np.isnan(y)]
    y_mod = y_mod[~np.isnan(y_mod)]
    n = len(y)
    if n == 0:
        return np.nan, n
    else:
        MSE = np.mean(((y - y_mod)**2))
        R2 = 1 - MSE/np.mean((y - y.mean())**2)
        # if (R2 > 1) | (R2 < 0): R2 = np.nan
        return R2, n

def calc_rmse(y, y_mod, debug=False):
    '''
    Fucntion to calculate the root mean square error
    
    Input:
    ------
    y (np.ndarray | xr.DataArray)               observed data
    y_mod (np.ndarray | xr.DataArray)           fitted data

    Output:
    -------
    RMSE (float)                                root mean square error

    Options:
    --------
    debug (bool)                                whether or not to print debug information
                                                default = False

    '''   
    assert len(y) == len(y_mod)
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    if isinstance(y, xr.DataArray): y = y.values.flatten()
    elif isinstance(y, np.ndarray): y = y.flatten()
    if isinstance(y_mod, xr.DataArray): y_mod = y_mod.values.flatten()
    elif isinstance(y_mod, np.ndarray): y_mod = y_mod.flatten()
    y[np.isnan(y_mod)] = np.nan
    y_mod[np.isnan(y)] = np.nan
    y = y[~np.isnan(y)]
    y_mod = y_mod[~np.isnan(y_mod)]
    n = len(y)
    if n == 0:
        return np.nan, n
    else:
        RMSE = np.sqrt(np.mean(((y - y_mod)**2)))
        return RMSE, n
    
def calc_rrmse(y, y_mod, debug=False):
    '''
    Fucntion to calculate the relative root mean square error
    
    Input:
    ------
    y (np.ndarray | xr.DataArray)               observed data
    y_mod (np.ndarray | xr.DataArray)           fitted data

    Output:
    -------
    RRMSE (float)                               relative root mean square error

    Options:
    --------
    debug (bool)                                whether or not to print debug information
                                                default = False

    '''   
    assert len(y) == len(y_mod)
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    if isinstance(y, xr.DataArray): y = y.values.flatten()
    elif isinstance(y, np.ndarray): y = y.flatten()
    if isinstance(y_mod, xr.DataArray): y_mod = y_mod.values.flatten()
    elif isinstance(y_mod, np.ndarray): y_mod = y_mod.flatten()
    y[np.isnan(y_mod)] = np.nan
    y_mod[np.isnan(y)] = np.nan
    y = y[~np.isnan(y)]
    y_mod = y_mod[~np.isnan(y_mod)]
    n = len(y)
    if n == 0:
        return np.nan, n
    else:
        RRMSE = np.sqrt(np.mean(((y - y_mod)**2)))/np.mean(y)
        return RRMSE, n

def is_leap_year(year, debug=False):
    '''
    Function to determine whether a year is a leap year
    Input:
    ------
    year (int)

    Output:
    ------
    leap_flag (bool)

    Options:
    --------
    debug (bool)            whether or not to print debug information
                            default = False
    '''

    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    if year % 400 == 0 or (year % 100 != 0 and year % 4 == 0):
        leap_flag = True
    else:
        leap_flag = False
    
    return leap_flag

## Load cropland weights
def load_cropland_weights(region='sub-national', axis='reg_code', soc='2015soc', debug=False):
    '''
    Function to load cropland weights

    Options:
    --------
    region (str)            regional aggregation, can choose from ['sub-national', 'national']
                            default = 'sub-national'
    axis (str)              name of the regional axis
                            default = 'reg_code'
    soc (str)               year of cropland weights, can choose from ['2015soc']
                            default = '2015soc'
    debug (bool)            whether or not to print debug information
                            default = False

    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    if region == 'sub-national':
        return xr.load_dataarray(f'{path_data}regions/cropland_area.nc')
    if region == 'national':
        wgt = xr.load_dataset(f'{path_data}regions/cropland_area.nc')
        wgt_na = aggreg_region(wgt, mod_region='National', old_axis=axis, new_axis=axis+'_new', debug=debug)
        for reg in wgt.coords[axis].values:
            if reg in wgt_na.coords[axis+'_new'].values:
                wgt.loc[{axis:reg}] = wgt_na.loc[{axis+'_new':reg}]
            else:
                wgt.loc[{axis:reg}] = 0
        wgt = wgt['weight']
        return wgt

def load_reg_coords(mod_region='Sub-national', dir=f'{path_data}regions/', file_suffix='crop', debug=False):
    '''
    Function to load coordinates for a given regional aggregation

    Output:
    -------
    coords (xr.Dataset)     dataset containing coordinates for the given regional aggregation

    Options:
    --------
    mod_region (str)        name of the regional aggregation, can choose from ['reg_code', 'National']
                            default = 'reg_code'
    dir (str)               directory of the coordinate files
                            default = f'{path_data}regions/'
    file_suffix (str)       suffix of the coordinate files, e.g., 'crop' for crop-related coordinates
                            default = 'crop'
    debug (bool)            whether or not to print debug information
                            default = False
    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    
    ## load regional codes
    with open(dir + f'OSCAR_reg_dict_{file_suffix}.csv', 'r') as f: TMP = np.array([line for line in csv.reader(f)])
    reg_code = TMP[1:, TMP[0, :].tolist().index(mod_region)].tolist()
    ## delete duplicates
    reg_code = list(set(reg_code))
    reg_code = [reg for reg in reg_code if reg != '' and reg.strip(' ') != '']
    reg_code.sort()
    ## move 'XXX' to the first of the list if exists
    if 'XXX' in reg_code:
        reg_code.remove('XXX')
        reg_code.insert(0, 'XXX')

    ## load long region names
    with open(dir + f'OSCAR_reg_names_{file_suffix}.csv') as f: 
        for line in csv.reader(f):
            if line[0] == mod_region: 
                long_name = line[1:]
                break
        else:
            raise KeyError(f'Long names for region "{mod_region}" not found.')
    long_name = {name_pair.split(':')[0]: name_pair.split(':')[1] for name_pair in long_name if name_pair != ''}
    print(f'Loaded long names for "{mod_region}": {len(long_name)} regions')
    assert len(long_name) == len(reg_code), 'The number of long names does not match the number of region codes.'
    assert all([reg in long_name.keys() for reg in reg_code]), 'Some long names are missing for the given region.'

    coords = xr.Dataset(coords={'reg_code': reg_code, 'long_name': ('reg_code', [long_name[reg] for reg in reg_code])})

    return coords

## identify timeseries outliers where values deviate by anomaly_threshold from rolling average.
def remove_timeseries_anomaly(da, window=5, anomaly_ratio=10, time_axis='year', debug=False):
    '''
    Function to detect and remove timeseries outliers
    
    Input:
    ------
    da (xr.DataArray)           timeseries data
    
    Output:
    -------
    da_new (xr.DataArray)       timeseries data without anomaly

    Options:
    --------
    window (int)                rolling window of the timeseries
                                default = 5
    anomaly_ratio (int)         data larger/lower than anomaly_ratio times of the rolling average are considered as outliers
                                default = 10
    time_axis (str)             name of the time axis
                                default = year
    debug (bool)                whether or not to print debug information
                                default = False             
    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    rolling_avg = da.rolling({time_axis: window}, center=True).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(da / rolling_avg)
    outliers = (ratio > anomaly_ratio) | (ratio < 1 / anomaly_ratio)
    da_new = da.where(~outliers)
    return da_new

## make sure of the consistency of a given dimension between two arrays
def sort_coords(ds1, ds2, axis = ['lat', 'lon'], debug=False):
    '''
    Function to sort the given dimensions of the first dataset based on the second dataset

    Input:
    ------
    ds1 (xr.Dataset)                    the first input dataset
    ds2 (xr.Dataset | xr.DataArray)     the second input dataset

    Output:
    ------
    ds_new (xr.Dataset)                 the new dataset

    Options:
    --------
    axis (list)                         the name of dimensions to be sorted
                                        default = ['lat', 'lon']
    debug (bool)                        whether or not to print debug information
                                        default = False
    
    '''

    assert all([dim in ds1.coords for dim in axis])
    assert all([dim in ds2.coords for dim in axis])
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    ds_new = ds1.copy(deep=True)
    for dim in axis:
        if any(ds1.coords[dim].values != ds2.coords[dim].values):
            ds_new = ds1.sortby(dim, ascending=True)
            if any(ds_new.coords[dim].values != ds2.coords[dim].values):
                ds_new = ds1.sortby(dim, ascending=False)
                if any(ds_new.coords[dim].values != ds2.coords[dim].values):
                    print(f'{dim} values among two datasets are not the same!')
                    raise RuntimeError

    return ds_new

def stack_dims(data, dims, new_dim, sep='_', how='all', debug=False):
    '''
    Function to stack dimensions into one dimension

    Input:
    ------
    data (xr.Dataset| xr.DataArray)     input data
    dims (list)                         list of dimensions to be stacked
    new_dim (str)                       name of the new dimension

    Output:
    -------
    ds (xr.Dataset| xr.DataArray)       data with stacked dimension

    Options:
    --------
    sep (str)                           seperator
                                        default = '_'
    debug (bool)                        whether or not to print debug information
                                        default = False

    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    try:
        from itertools import starmap
    except:
        raise ImportError("'itertools' libraries must be installed")
    
    ds = data.copy(deep=True)
    ds = ds.stack({new_dim: dims})
    fstr = sep.join(['{}'] * ds.indexes[new_dim].nlevels)
    idx = ds.indexes[new_dim]
    ds = ds.reset_index(new_dim)
    ds[new_dim] = list(starmap(fstr.format, idx))
    ds = ds.dropna(dim=new_dim, how=how)
    return ds

def trans_tif_grid(filename, center=True, debug=False):
    '''
    Function to transform tiff grid to georeferenced latitude and longitude
    ## reference: https://gdal.org/tutorials/geotransforms_tut.html

    Input:
    ------
    filename (str)      name of the tiff file
    
    Output:
    -------
    lat (np.array)      latitude
    lon (np.array)      longitude

    Options:
    --------
    center (bool)       whether or not to use the center coordiante
                        default = True
    debug (bool)        whether or not to print debug information
                        default = False

    '''
    try:
        from osgeo import gdal
    except:
        raise ImportError("'gdal' libraries must be installed")
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    ds = gdal.Open(filename)
    adfGeoTransform = ds.GetGeoTransform()
    nXSize = ds.RasterXSize
    nYSize = ds.RasterYSize
    lon = np.zeros(nXSize)
    lat = np.zeros(nYSize)
    for i in np.arange(nXSize):
        for j in np.arange(nYSize):
            lon[i] = adfGeoTransform[0] + i * adfGeoTransform[1] + j * adfGeoTransform[2]
            lat[j] = adfGeoTransform[3] + i * adfGeoTransform[4] + j * adfGeoTransform[5]
    if center:
        lon = lon + adfGeoTransform[1]*0.5
        lat = lat + adfGeoTransform[5]*0.5
    return lat, lon

## convert crop specifier to cropland type
def convert_crop_land(specifier, irr, debug=False):
    '''
    Function to convert ISIMIP crop specifier to cropland type

    Input:
    ------
    specifier (str)         crop specifier
    irr (str)               irrigation
                            default = 'noirr'
    
    Output:
    ------
    var (list)              name of cropland type

    Options:
    --------
    debug (bool)            whether or not to print debug information
                            default = False
    
    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    LU_type = {'noirr':{'mai':['maize_rainfed'],
        'soy':['oil_crops_soybean_rainfed'],
        'ric':['rice_rainfed'],
        'ri1':['rice_rainfed'],
        'ri2':['rice_rainfed'],
        'whe':['temperate_cereals_rainfed'],
        'swh':['temperate_cereals_rainfed'],
        'wwh':['temperate_cereals_rainfed'],
        'euc':['c3per_rainfed_bf'],
        'mis':['c4per_rainfed_bf'],
        'pop':['c3per_rainfed_bf']},
        'firr':{'mai':['maize_irrigated'],
        'soy':['oil_crops_soybean_irrigated'],
        'ric':['rice_irrigated'],
        'ri1':['rice_irrigated'],
        'ri2':['rice_irrigated'],
        'whe':['temperate_cereals_irrigated'],
        'swh':['temperate_cereals_irrigated'],
        'wwh':['temperate_cereals_irrigated'],
        'euc':['c3per_irrigated_bf'],
        'mis':['c4per_irrigated_bf'],
        'pop':['c3per_irrigated_bf']
        }}
    
    return LU_type[irr][specifier]

##################################################
##  DISPLAY DATA
##################################################
def print_extrema(data: xr.DataArray, n: int = 10, method: str = 'max', debug=False):
    '''
    Print top N min/max values with coordinates for inspection
    
    Input:
    ------
    data (xr.DataArray)         input data (any dimensions)

    Options:
    -------
    n (int)                     number of extrema to print
                                default = 10
    method (str)                'max' or 'min'
                                default = 'max'
    debug (bool)                whether or not to print debug information
                                default = False
    '''
    # Validate input data type
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Data must be numeric (got {data.dtype})")
    
    try:
        # handle NaN values
        flat = data.stack(z=data.dims).dropna('z')
        
        # sort data
        if method == 'max':
            sorted_data = flat.sortby(flat, ascending=False)
            topn = sorted_data.isel(z=slice(0, n))
        elif method == 'min':
            sorted_data = flat.sortby(flat, ascending=True)
            topn = sorted_data.isel(z=slice(0, n))
        else:
            raise ValueError("method must be either 'max' or 'min'")
        
        # print header
        print(f'Top {n} {method} values:')
        print('-' * 40)
        
        # print results
        for i in range(min(n, len(topn.z))):
            # get coordinates
            coords = {dim: topn[dim].isel(z=i).values.item() 
                     for dim in data.dims}
            
            # format coordinate string
            coord_str = ', '.join(
                f'{k}: {v:.3f}' if isinstance(v, (float, np.floating))
                else f'{k}: {v}' 
                for k, v in coords.items()
            )
            
            # get and format value
            value = topn.isel(z=i).item()
            print(f'{i+1}. {coord_str} | Value: {value:.5f}')
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

def retrieve_values_by_threshold(data: xr.DataArray, above: float = None, below: float = None, output=None, display=True, debug=False):
    '''
    Print coordinates where data values exceed an upper threshold OR fall below a lower threshold,
    with proper formatting for different coordinate types.

    Input:
    ------
    data (xr.DataArray)         input data array

    Options:
    -------
    above (float)               upper threshold value
                                default = None
    below (float)               lower threshold value
                                default = None
    output (str)                path to output file (if None, prints to console)
                                default = None
    display (bool)              whether or not to print results to console
                                default = True
    debug (bool)                whether or not to print debug information
                                default = False
    '''
    # find all points above or below threshold
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    if output is not None: assert isinstance(output, str)

    mask_above = data > above if above is not None else False
    mask_below = data < below if below is not None else False
    if above is not None and below is not None:
        if above > below: 
            mask = mask_above | mask_below
        else:  # above <= below
            mask = mask_above & mask_below
            print(f'Coordinates where: {above} < values < {below}:')
    elif above is not None and below is None:
        mask = mask_above
        print(f'Coordinates where: values > {above}')
    elif above is None and below is not None:
        mask = mask_below
        print(f'Coordinates where: values < {below}')
    else:
        print('No thresholds specified. Returning original data.')
        raise ValueError('No thresholds specified.')
    data_ma = data.where(mask, drop=True)

    # get coordinates for each point
    stacked = data_ma.stack(point=data.dims).dropna('point')

    # print results with formatted coordinates
    if above is None and below is None:
        print('No thresholds specified. Returning original data.')
        print(data)
        return

    print('=' * 50)
    if len(stacked.point) == 0:
        print('No data points found outside the specified thresholds.')
    else:
        if output is not None:
            with open(output, 'w') as f:
                f.write(f"{', '.join(data.dims)}, value\n")
        else:
            if display:
                print(f"{', '.join(data.dims)}, value")
        for i in range(len(stacked.point)):
            value = stacked.isel(point=i).values.item()
            # get all coordinate values
            coords = {dim: stacked[dim].isel(point=i).values.item()
                    for dim in data.dims}

            # format based on coordinate type
            coord_str = []
            for dim, val in coords.items():
                if isinstance(val, (float, np.floating)):
                    coord_str.append(f"{val:.3f}")
                else:
                    coord_str.append(f"{val}")

            if output is not None:
                with open(output, 'a') as f:
                    f.write(f"{', '.join(coord_str)}, {value:.5f}\n")
            else:
                if display:
                    print(f"{', '.join(coord_str)}, {value:.5f}")

    print(f'Found {len(stacked.point)} points outside the thresholds')
    return stacked

##################################################
##  FIND AND LOAD DATA
##################################################
def find_files_isimip3b_var(var, dir, gcms=None, scens=None, keys=None, debug=False):
    '''
    Find ISIMIP3b-related files and return name list

    Input:
    ------
    var (str)               variable (e.g., 'tas', 'pr')
    dir (str)               path to ISIMIP3b data

    Output:
    -------
    files_out (list)        list of paths to all the filenames found for the given input

    Options:
    --------
    gcms (list)             Earth System Models (e.g., 'CanESM2' or 'CanESM5')
                            default = None
    scens (list)            a list of scenarios (e.g., ['ssp126' or 'ssp585'])
                            default = None
    keys (list)             a list of strings that must be included in the file names
                            default = None
    debug (bool)            whether or not to print debug information
                            default = False

    '''
    class NotFound(Exception):
        pass

    assert (isinstance(gcms, list) if gcms is not None else True)
    assert (isinstance(scens, list) if scens is not None else True)
    assert (isinstance(keys, list) if keys is not None else True)

    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    ## change all the input words into their lower case
    if gcms is not None: gcms = [gcm.lower() for gcm in gcms]
    if scens is not None: scens = [scen.lower() for scen in scens]
    if keys is not None: keys = [key.lower() for key in keys]

    files_out = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            fn = file.lower()
            try:
                if var not in fn: raise NotFound

                ## check whether the file contains all the necessary keys
                flag_key = True
                if keys is not None:
                    flag_key = all([key in fn for key in keys])
                    if not flag_key: raise NotFound

                ## check whether the file contains one of the given gcms
                flag_gcm = True
                if gcms is not None:
                    flag_gcm = any([gcm in fn for gcm in gcms])
                if not flag_gcm: raise NotFound

                ## check whether the file contains one of the given scens
                flag_scen = True
                if scens is not None:
                    flag_scen = any([scen in fn for scen in scens])
                if not flag_scen: raise NotFound

            except NotFound:
                continue

            else:
                files_out.append(os.path.join(root, file))

    if len(files_out) == 0:
        print(f'{var} is not in {dir} and its sub-directories.')
        raise RuntimeError

    return files_out

##################################################
##  GROWING SEASON
##################################################
## convert from day of year to another date format
def convert_doy(days_in, year, format='dd', debug=False):
    '''
    Function to convert day of year to another date format
    Input:
    ------
    days_in (np.ndarray)    array of day of year
    year (int)              year
    
    Output:
    ------
    days (np.ndarray)       new array of given date format
    
    Options:
    ------
    format (str)            date format, can choose from ['dd', 'mm']
                            default = 'dd'
    debug (bool)            whether or not to print debug information
                            default = False
    
    '''

    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    days = []
    for i in np.arange(len(days_in.flatten())):
        if days_in.flatten()[i] > 0:
            if format == 'mm':
                days.append(np.datetime64(datetime.strptime(str(year)+str(days_in.flatten()[i]), '%Y%j'), 'M'))
            if format == 'dd':
                days.append(np.datetime64(datetime.strptime(str(year)+str(days_in.flatten()[i]), '%Y%j'), 'D'))
        else:
            days.append(np.datetime64('NaT'))

    days = np.array(days, dtype='datetime64').reshape(days_in.shape)

    return days

## transfrom the growing season time format
def trans_doy_gs(day_p, day_m, year, method = 'forward', time_scale = 'day', debug=False):
    '''
    Function to transform the crop calendar format from day of year to np.datetime64
    
    Input:
    ------
    day_p (np.ndarray)      input array of day of planting
    day_m (np.ndarray)      input array of day of maturity
    lat (np.ndarray)        input array of latitude
    lon (np.nadrray)        input array of longitude

    Output:
    ------
    dop (np.ndarray)        time of planting
    dom (np.ndarray)        time of maturity
    
    Options:
    ------
    method (str)            how to deal with growing season that is not within the same calendar year
                            default = 'forward'
                            another option is 'backward'
    time_scale (str)        time resolution of output weight, can choose from ['month', 'day']
                            default = 'day'
    debug (bool)            whether or not to print debug information
                            default = False
    '''

    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    assert isinstance(day_p, np.ndarray) and isinstance(day_m, np.ndarray)
    time_s = time.time()

    dop = np.nan * np.zeros_like(day_p)
    dom = np.nan * np.zeros_like(day_m)
    if time_scale == 'month':
        ## convert day of year to the month of year
        dop = convert_doy(day_p, year, format='mm')
        dom = convert_doy(day_m, year, format='mm')
        if method == 'forward': dom = np.where(dop > dom, dom+np.timedelta64(12, 'M'), dom)
        elif method == 'backward': dop = np.where(dop > dom, dop-np.timedelta64(12, 'M'), dop)
    elif time_scale == 'day':
        dop = convert_doy(day_p, year, format='dd')
        dom = convert_doy(day_m, year, format='dd')
        if is_leap_year(year):
            if method == 'forward': dom = np.where(dop > dom, dom+np.timedelta64(366, 'D'), dom)
            elif method == 'backward': dop = np.where(dop > dom, dop-np.timedelta64(366, 'D'), dop)
        else:
            if method == 'forward': dom = np.where(dop > dom, dom+np.timedelta64(365, 'D'), dom)
            elif method == 'backward': dop = np.where(dop > dom, dop-np.timedelta64(365, 'D'), dop)

    time_e = time.time()
    time_r = time_e - time_s
    print("Time consumed: {:.0f}min {:.0f}sec".format(time_r // 60, time_r % 60))

    return dop, dom

##################################################
##  REGIONAL AGGREGATION & DISAGGREGATION
##################################################
def aggreg_grid(ds_in, mask, 
        weight=None, 
        calc_global=False, 
        weight_scheme='area', 
        old_axis=['lat', 'lon'], 
        new_axis='reg_code', 
        time_axis='year', 
        weight_output=True, 
        weight_var='weight', 
        method='mean',
        chunk_time=True,
        target_MB=32,
        debug=False
    ):
    '''
    Aggregate grid data to OSCAR regions based on given mask file
    
    Input:
    ------
    ds_in (xr.Dataset)                  input dataset to be aggregated

    mask (xr.Dataset)                   regional mask dataset
                                        must contain the following variables:
                                        - new_axis: regional code (e.g., 'reg_code')
    Output:
    -------
    ds_out (xr.Dataset)                 output dataset

    Options:
    --------
    calc_global (bool)                  whether to calculate the global average (considering all grid cells)
                                        the calculation will be independent of regional masks
                                        default = True
    weight_scheme (str)                 scheme of applying weight, choose from ['area', 'one']
                                        this is necessary for intensive variables (e.g. temperature that needs to be weighted by area); 
                                        default = 'area'
    weight (xr.DataArray)               additional weight variable; 
                                        default = None
    old_axis (list)                     name of regional axis that will be aggregated (must a dim of ds_in);
                                        default = ['lat', 'lon']
    new_axis (str)                      name of new aggregated regional axis (must NOT be in ds_in, and will be in ds_out);
                                        default = 'reg_code'
    time_axis (str)                     name of time axis (to ensure it is first dim in ds_out);
                                        default = 'year'
    weight_output (bool)                whether to output weight of the aggregated mask
                                        default = True
    weight_var (str)                    name of the output weight variable
                                        default = 'weight'
    method (str)                        calculation method of data, valid methods include 'mean' and 'sum'
                                        default = 'mean'
    chunk_time (bool)                   whether to chunk the data by time
                                        default = True
    target_MB (float)                   target MB for chunking when chunk_by_time is True
                                        default = 32
    debug (bool)                        whether or not to print debug information
                                        default = False
    
    '''
    warnings.filterwarnings('ignore')
    
    ## check old axis in ds_in and new_axis not in ds_in
    assert all(axis in ds_in.coords for axis in old_axis) and new_axis not in ds_in.coords
    assert isinstance(mask, xr.Dataset)
    if weight is not None: assert isinstance(weight, xr.DataArray)
    if calc_global: assert 'area' in mask.data_vars, 'area variable must be provided in the mask dataset for global calculation'
    
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    t0 = time.time()

    ## prepare regional mask
    mask = sort_coords(mask, ds_in, axis=old_axis)
    mask_reg = mask['frac_reg']
    reg_code = mask[new_axis].values

    if debug:
        print(f'Number of regions in mask: {len(reg_code)}')

    ## prepare weight (could be time-varying)
    if weight_scheme == 'area': 
        assert 'area' in mask.data_vars, 'area variable must be provided in the mask dataset for area weighting'
        assert mask['area'].attrs['units'] == 'm2', 'unit of area variable in mask file must be m2'
        area = mask['area'] * 1.0e-6  # from m2 to km2
    elif weight_scheme == 'one': area = xr.ones_like(ds_in.lat)
    else: raise ValueError('weight_scheme must be either "area" or "one", got {}'.format(weight_scheme))

    ## extract variables without regional axis
    vars_with_old_axis = [var for var in ds_in if (all(axis in ds_in[var].dims for axis in old_axis))]
    vars_without_old_axis = [var for var in ds_in if (any(axis not in ds_in[var].dims for axis in old_axis))]

    if debug:
        print(f'Variables with all old axes ({old_axis}): {vars_with_old_axis}')
        print(f'Variables without all old axes: {vars_without_old_axis}')

    ds_out = xr.Dataset(coords={new_axis: reg_code})

    for var in vars_without_old_axis:
        if var not in old_axis:
            ds_out[var] = ds_in[var]
            if debug: print(f'Variable "{var}" does not contain all old axes, copied directly to output.')

    ## prepare time chunking
    has_time = time_axis in ds_in.dims

    if chunk_time and has_time:
        # get number of time steps
        n_time = len(ds_in[time_axis])
        
        # calculate memory per time step
        spatial_cells = 1
        for axis in old_axis:
            if axis in ds_in.dims:
                spatial_cells *= len(ds_in[axis])
        
        bytes_per_step = ds_in[vars_with_old_axis[0]].nbytes * np.prod([len(mask[dim]) for dim in mask.dims if dim not in old_axis])
        MB_per_step = bytes_per_step / 1024**2
        
        # determine steps per chunk based on memory target
        if MB_per_step > 0:
            steps_per_chunk = max(1, int(target_MB / (MB_per_step)))
        else:
            steps_per_chunk = n_time
        
        # ensure chunk size is reasonable
        steps_per_chunk = min(steps_per_chunk, n_time)
        
        # for very small datasets, use one chunk
        if n_time <= steps_per_chunk:
            steps_per_chunk = n_time

        ds_in = ds_in.chunk({time_axis: steps_per_chunk})
        
        if debug:
            print(f'Time dimension: {time_axis}')
            print(f'Total time steps: {n_time}')
            print(f'Memory per step: {MB_per_step:.6f} MB')
            print(f'Target per chunk: {target_MB:.2f} MB')
            print(f'Steps per chunk: {steps_per_chunk}')
            print(f'Actual chunk memory: {steps_per_chunk * MB_per_step:.4f} MB')
            print(f'Number of chunks: {np.ceil(n_time / steps_per_chunk):.0f}')
        
        if debug:
            if hasattr(ds_in[time_axis].data, 'nparticles'):
                print(f'Created {ds_in[time_axis].data.nparticles} chunks')
    
    wgt = area * mask_reg * weight if weight is not None else area * mask_reg
    wgt = wgt.stack(cell=old_axis)

    ## time chunking and aggregation
    for var in vars_with_old_axis:
        var_data = ds_in[var].stack(cell=old_axis)
        norm = xr.dot(var_data.notnull(), wgt, dim='cell')
        with ProgressBar():
            da_out = xr.dot(var_data.fillna(0), wgt, dim='cell').compute()

        ds_out[var] = da_out
        if method == 'mean': ds_out[var] = ds_out[var] / norm

        if calc_global:
            weight_g = xr.dot(var_data.notnull(), area.stack(cell=old_axis), dim='cell')
            var_global = xr.dot(var_data.fillna(0), area.stack(cell=old_axis), dim='cell').rename(f'{var}_g')
            if method == 'mean': var_global /= weight_g
            ds_out[f'{var}_g'] = var_global
            if 'long_name' in ds_in[var].attrs: ds_out[f'{var}_g'].attrs['long_name'] = f"{ds_in[var].attrs['long_name']} (global)"

            gc.collect()

        if debug:
            print(f'Output variable "{var}" shape: {ds_out[var].shape}, dims: {ds_out[var].dims}')

    ## add weight variable
    if weight_output:
        ds_out[weight_var] = wgt.sum('cell')
        ds_out[weight_var] = ds_out[weight_var].assign_attrs({'units': 'km^2'})
        ds_out[weight_var+'_g'] = weight_g
        if debug: print(f'Output weight variable "{weight_var}" shape: {ds_out[weight_var].shape}, dims: {ds_out[weight_var].dims}')

    ## add mask information for OSCAR regional aggregation
    if 'reg_name' in mask:
        ds_out = ds_out.assign(reg_name=mask['reg_name'])
        ds_out = ds_out.set_coords('reg_name')

    # make sure time axis is first
    if time_axis in ds_out.coords: 
        ds_out = ds_out.transpose(time_axis,...)

    ds_out.attrs['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    t = time.time() - t0
    if debug: 
        print(f'\n{"="*60}')
        print(f'AGGREGATION COMPLETE')
        print(f'{"="*60}')
        print(f'Total time: {t//60:.0f}min {t%60:.0f}sec')
        print(f'Output dataset size: {ds_out.nbytes / 1024**2:.2f} MB')
        print(f'Output variables: {list(ds_out.data_vars)}')
        print(f'Output dimensions: {ds_out.dims}')
    return ds_out

        
## aggregate regional data to OSCAR regions
## /!\ WARNING: translated regions that are not in input data will not appear in output data (i.e. might have to use combine_first instead of merge, with the output data)
def aggreg_region(ds_in, new_region,  
    weight_dict={}, 
    old_region='Sub-national', 
    old_axis='reg_code', 
    new_axis='reg_code_new', 
    time_axis='year', 
    dir=f'{path_data}regions/',
    file_suffix='crop',
    debug=False):
    '''
    Function to aggregate data onto OSCAR regions. It uses dictionnaries mapping ISO regions to OSCAR regions defined in 'input_data/regions' by user.
    
    Input:
    ------
    ds_in (xr.Dataset)  input dataset to be aggregated
    new_region (str)    name of regional aggregation (must be a valid option)
        
    Output:
    -------
    ds_out (xr.Dataset) output dataset

    Options:
    --------
    weight_dict (dict)  keys variables are weighted using values variables when aggregating; 
                        this is necessary for intensive variables (e.g. temperature that needs to be weighted by area); 
                        keys and values are names (str) of ds_in variables;
                        keys are variables to be weighted, values are weight variables;
                        default = {}
    old_region (str)    name of regional axis that will be aggregated (must a dim of ds_in);
                        default = 'reg_code'
    old_axis (str)      name of regional axis that will be aggregated (must a dim of ds_in);
                        default = 'reg_code'
    new_axis (str)      name of new aggregated regional axis (must NOT be in ds_in, and will be in ds_out);
                        default = 'reg_code_new'
    time_axis (str)     name of time axis (to ensure it is first dim in ds_out);
                        default = 'year'
    dir (str)           path to directory containing regional mapping files;
                        default = f'{path_data}regions/'
    debug (bool)        whether or not to print debug information
                        default = False
    '''
   
    ## check old axis in ds_in and new_axis not in ds_in
    assert old_axis in ds_in.coords and new_axis not in ds_in.coords

    ## check all weight variables in ds_in
    for key, val in weight_dict.items():
        if key not in ds_in.data_vars:
            raise KeyError(f'Weight variable "{key}" not found in dataset.')
        if val not in ds_in.data_vars:
            if val == 1:
                print(f'Sum up variable "{key}" in the dataset')
            else:
                raise KeyError(f'Weight variable "{val}" not found in dataset.')

    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    warnings.filterwarnings('ignore')

    ## make deep copy to be safe
    ds_out = ds_in.copy(deep=True)

    ## region mapping files to be loaded
    list_load = [zou for zou in os.listdir(dir) if all([_  in zou for _ in ['dict', '.csv']])]

    ## load and create combined dictionary
    dico = {}
    for zou in list_load:
        with open(dir + zou) as f: TMP = np.array([line for line in csv.reader(f)])
        if old_region in TMP[0,:].tolist() and new_region in TMP[0,:].tolist():
            dico = {**dico, **{key:val for key, val in zip(TMP[1:,TMP[0,:].tolist().index(old_region)], TMP[1:,TMP[0,:].tolist().index(new_region)])}}
    if debug: print(f'Region mapping file for old region "{old_region}" and new region "{new_region}" in {dir}.')

    values_empty = [key for key, val in dico.items() if val == '' or val.strip() == '']
    for key in values_empty:
        if key == '' or key.strip() == '':
            pass
        else:
            print(f'Caution: Empty value found in region mapping file for {key}.')
        dico.pop(key)

    keys_empty = [key for key, val in dico.items() if key == '' or key.strip() == '']
    for key in keys_empty:
        if dico[key] == '' or dico[key].strip() == '':
            pass
        else:
            print(f'Non-empty value {dico[key]} found in {new_region}')
        dico.pop(key)

    ## load long region names
    with open(dir+'OSCAR_reg_names_{}.csv'.format(file_suffix)) as f: 
        for line in csv.reader(f):
            if line[0] == new_region: 
                long_name = line[1:]
                break
        else:
            raise KeyError(f'Long names for region "{new_region}" not found.')
    long_name = {name_pair.split(':')[0]: name_pair.split(':')[1] for name_pair in long_name if name_pair != ''}
    assert all([reg in long_name for reg in dico.values()]), 'Some long names are missing for the given region.'

    ## apply weights to weighted variables
    for key, val in weight_dict.items():
        if val != 1:
            ## deal with nan values in weight variable
            ds_out[val+'_mask'] = ds_out[key].notnull()
            ds_out[key+'_'+val] = (ds_out[val] * ds_out[val+'_mask']).fillna(0)
            ds_out[key] = ds_out[key] * ds_out[key+'_'+val]
        else:
            ds_out[key] = ds_out[key]
    if val != 1:
        for val in weight_dict.values():
            if val in ds_out.data_vars: ds_out = ds_out.drop_vars([val])
            if val+'_mask' in ds_out.data_vars: ds_out = ds_out.drop_vars([val+'_mask'])

    ## extract variables without regional axis
    ds_non = ds_out.drop([var for var in ds_out if old_axis in ds_out[var].dims] + [old_axis])
    ds_out = ds_out.drop([var for var in ds_out if old_axis not in ds_out[var].dims])

    ## drop regional axis values that are not in the mapping file
    ## ! this is a very coarse way to do it
    keys_drop = [reg for reg in ds_out[old_axis].values if reg not in dico.keys()]
    if len(keys_drop) > 0:
        print(f'Warning: The following regions in the dataset are not in the mapping file and will be dropped:\n{keys_drop}')
        ds_out = ds_out.drop_sel({old_axis: keys_drop})

    ## new regional aggregation
    ds_out.coords[new_axis] = xr.DataArray([dico[reg] for reg in ds_out[old_axis].values], dims=old_axis)
    ds_out = ds_out.groupby(new_axis).sum(old_axis, keep_attrs=True, min_count=1)
    ds_out.coords[new_region + '_name'] = xr.DataArray([long_name[reg] for reg in ds_out[new_axis].values], dims=new_axis)

    ## scaled by weights
    for key, val in weight_dict.items():
        if val != 1:
            ds_out[key] = xr.where(ds_out[key+'_'+val] != 0, ds_out[key] / ds_out[key+'_'+val], np.nan)
        else:
            ds_out[key] = ds_out[key]
    
    ## merge with extracted variables
    ds_out = xr.merge([ds_out, ds_non])

    ## make sure no variable with old_axis
    assert all([old_axis not in ds_out[var].dims for var in ds_out.data_vars]), f'Some variables still have the old axis "{old_axis}" after aggregation.'
    try:
        ds_out = ds_out.drop_dims(old_axis)
    except:
        pass

    ## make sure time axis is first
    if time_axis in ds_out.coords: 
        ds_out = ds_out.transpose(time_axis,...)
    
    ds_out = ds_out.sortby(new_axis)
    
    ## return
    return ds_out

## split region into disaggreated regions
def convert_reg_code(reg, region_from='Sub-national', region_to='National', debug=False):
    '''
    Function to convert between different region codes
    Input:
    ------
    reg (str)           region to be disaggregated
                        
    Output:
    ------
    reg_new (list)      new list of regional code
    
    Options:
    --------
    region_from (str)   name of regional aggregation
                        default = 'Sub-national'
    region_to (str)     name of regional aggregation
                        default = 'National'
    debug (bool)        whether or not to print debug information
                        default = False
    '''
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')

    ## region mapping files to be loaded
    list_load = [zou for zou in os.listdir(f'{path_data}regions/') if all([_  in zou for _ in ['dict', '.csv']])]
    
    ## load and create combined dictionary
    dico = {}
    for zou in list_load:
        with open(f'{path_data}regions/' + zou) as f: TMP = np.array([line for line in csv.reader(f)])
        if region_from in TMP[0,:].tolist() and region_to in TMP[0,:].tolist():
            for key, val in zip(TMP[1:,TMP[0,:].tolist().index(region_from)], TMP[1:,TMP[0,:].tolist().index(region_to)]):
                if reg == key: dico.setdefault(reg, []).append(val)
    try:
        reg_new = sorted(set(dico[reg]))
    except KeyError:
        print(f'Region {reg} not found in {region_from} to {region_to} mapping file.')
        return []
        
    return reg_new

##################################################
##  REGIONAL PLOT
##################################################
## create global map for regional data
def plot_global_map(var_in, levels,
        mask=None,
        axis='reg_code',
        crs=None,
        map_extent=[-180, 180, -90, 90],
        ax=None,
        title=None,
        draw_labels=False,
        cb_on=True,
        axis_label=['left', 'bottom'],
        contourf_kwargs={},
        colorbar_kwargs={},
        debug=False):
    '''
    Function to create a global map of a given variable
    
    Input:
    ------
    var_in (xr.DataArray)       1-D array, containing regional values to be plotted
    levels (np.array)           levels of contour map
        
    Output:
    -------
    ax (mpl.axes._axes.Axes)    axes containing the plot
    cf (QuadContourSet)         contour set of the plot

    Options:
    --------
    mask (xr.DataArray)         mask dataarray
                                default = None
    axis (str)                  regional axis
                                default = 'reg_code'
    region (str)                regional level
                                default = 'sub-national'
    crs (cartopy.crs)           coordinate reference system for the plot
                                default = None
    map_extent (list)           map extent in the form [lon_min, lon_max, lat_min, lat_max]
                                default = [-180, 180, -90, 90]
    ax (mpl.axes._axes.Axes)    axes to draw plot on
                                default = None
    title (str)                 title of the plot
                                default = None
    draw_labels (boolean)       whether to draw grid labels
                                default = False
    cb_on (boolean)             whether to draw colorbar
                                default = False
    contourf_kwargs             keyword arguments control the contour plot
                                default = {}
    colorbar_kwargs             keyword arguments control the colorbar
                                default = {}
    debug (bool)                whether or not to print debug information
                                default = False
    '''
    
    if debug: print(f'>>> Running {sys._getframe().f_code.co_name} <<<')
    warnings.filterwarnings('ignore')

    ## check old axis in ds_in and new_axis not in ds_in
    assert var_in.ndim == 1 and len(var_in[axis]) == len(var_in), 'Input data must be 1-D array with length equal to the length of the given axis.'
    
    if mask is None:
        print('Please load mask dataarray before plotting regional data.')
        raise RuntimeError
    
    var = sum([np.nan * xr.zeros_like(mask.coords[dim], dtype=float) for dim in ['lat', 'lon']])
    for reg in var_in.coords[axis]:
        if var_in.loc[{axis:reg.item()}].notnull().sum() > 0:
            try:
                if axis != 'reg_code': reg_list = convert_reg_code(reg.item(), region_from=axis, region_to='reg_code', debug=debug)
                for reg_sub in (reg_list if axis != 'reg_code' else [reg]):
                    var = xr.where(mask.loc[{'reg_code': reg_sub}] > 0, var_in.loc[{axis:reg.item()}].values, var)
            except KeyError:
                continue
    
    var = var.fillna(np.nan)
    if var.notnull().sum() == 0:
        print('No valid data to plot.')
        return None, None
    else:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import ticker
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        except ImportError:
            print('"cartopy" libraries must be installed')
            return None, None
        finally:
            if debug: print('Plotting contour map ...')
            if ax is None:
                ax = plt.subplot(111, projection=crs if crs is not None else ccrs.PlateCarree(central_longitude=0.0))
            if debug: print(f'Using projection: {type(ax.projection)}')
            ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#E6F7FF')
            ax.add_feature(ocean, zorder=0)
            
            if title is not None: ax.set_title(title)
            cf = ax.contourf(var.lon, var.lat, var, levels, transform=ccrs.PlateCarree(), zorder=4, **contourf_kwargs)

            ax.add_feature(cfeature.BORDERS, zorder=4, linewidth=0.15)
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), zorder=4, linewidth=0.15)
            
            if cb_on:
                cb_defaults = {'orientation': 'horizontal', 'aspect': 30, 
                            'shrink': 0.8, 'pad': 0.08, 'extend': 'both'}
                cb_defaults.update(colorbar_kwargs)
                cb = plt.colorbar(cf, **cb_defaults)

            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
            
            is_polar = isinstance(crs, (ccrs.NorthPolarStereo, ccrs.SouthPolarStereo))
            if is_polar:
                if draw_labels:
                    gl = ax.gridlines(
                        draw_labels=True, dms=True,
                        linewidth=0.5, color='gray', alpha=0.5, 
                        linestyle='--', zorder=3
                    )
                    
                    # configure label positions for polar projection
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.left_labels = True
                    gl.bottom_labels = True
                    
                    # set formatters
                    gl.xformatter = LongitudeFormatter(zero_direction_label=False)
                    gl.yformatter = LatitudeFormatter()
                    
                    # style the labels
                    gl.xlabel_style = {'size': 'small', 'color': 'black'}
                    gl.ylabel_style = {'size': 'small', 'color': 'black'}
                
            if not is_polar:
                try:
                    ax.set_xticks(np.arange(-180, 180, 60), crs=ccrs.PlateCarree())
                    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
                except RuntimeError:
                    pass
                try:
                    ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
                    ax.yaxis.set_major_formatter(LatitudeFormatter())
                except RuntimeError:
                    pass
                
                # add gridlines with labels for non-polar
                if draw_labels:
                    gl = ax.gridlines(draw_labels=draw_labels, linewidth=0.5, 
                                    color='gray', alpha=0.5, zorder=3)
                    gl.ylocator = ticker.FixedLocator([val for val in np.arange(-60, 90, 30) 
                                                    if val > map_extent[2] and val < map_extent[3]])
                    gl.yformatter = LatitudeFormatter()
                    gl.xformatter = LongitudeFormatter(zero_direction_label=False)


            if not is_polar and len(axis_label) > 0:
                ax.spines[['left', 'right', 'top', 'bottom']].set_linewidth(0.5)
                for side in axis_label:
                    ax.spines[side].set_visible(True)

            if not is_polar:
                ax.tick_params(
                    left=False if 'left' not in axis_label else True, 
                    right=False if 'right' not in axis_label else True, 
                    top=False if 'top' not in axis_label else True, 
                    bottom=False if 'bottom' not in axis_label else True,
                    labelleft=True if 'left' in axis_label else False,
                    labelright=True if 'right' in axis_label else False,
                    labeltop=True if 'top' in axis_label else False,
                    labelbottom=True if 'bottom' in axis_label else False
                )
            else:
                # for polar projections, turn off all default tick labels
                ax.tick_params(
                    left=False, right=False, top=False, bottom=False,
                    labelleft=False, labelright=False, labeltop=False, labelbottom=False
                )
            return ax, cf