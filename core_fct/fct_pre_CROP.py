##################################################
##################################################
import os
import csv
import sys
import glob
import time
import datetime 
import warnings
import numpy as np
import xarray as xr
from datetime import datetime

def is_leap(year):
    '''
    Fucntion to determine whether a year is a leap year
    Input:
    ------
    year (int)

    Output:
    ------
    leap_flag (bool)
    '''

    if year%400 == 0 or (year%100 != 0 and year%4 == 0):
        leap_flag = True
    else:
        leap_flag = False
    
    return leap_flag

## https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
## reference: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
def calc_earth_cellarea(lat_mid, dlat, dlon):
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
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

    # equatorial radius (km)
    a_Earth = 6378.137
    # polar radius (km)
    b_Earth = 6356.752
    # square of first eccentricity
    e2 = (a_Earth**2-b_Earth**2)/a_Earth**2
    # grid cell area (km2)
    area = a_Earth**2*(1-e2)/(1-e2*np.sin(np.radians(lat_mid))**2)**2*np.radians(dlat)*np.radians(dlon)*np.cos(np.radians(lat_mid))
    return area

## make sure of the consistency of a given dimension between two arrays
def sort_coords(ds1, ds2, 
        axis = ['lat', 'lon']):
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
    
    '''

    assert all([dim in ds1.coords for dim in axis])
    assert all([dim in ds2.coords for dim in axis])
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

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

def stack_dims(data, dims, new_dim, sep='_', how='all'):
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

    '''
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')
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

def calc_bic(y, y_mod, p):
    '''
    Fucntion to calculate the Bayesian information criterion
    
    Input:
    ------
    y (np.array)                observed data
    y_mod (np.array)            fitted data
    p (int)                     number of parameters

    Output:
    -------
    BIC (float)                 Bayesian information criterion

    '''
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')
    
    assert len(y) == len(y_mod)
    y = y.values.flatten()
    y_mod = y_mod.values.flatten()
    y[np.isnan(y_mod)] = np.nan
    y_mod[np.isnan(y)] = np.nan
    y = y[~np.isnan(y)]
    y_mod = y_mod[~np.isnan(y_mod)]
    n = len(y)
    SSE = np.sum(((y - y_mod)**2))
    BIC = n*np.log(SSE/n) + p*np.log(n)
    return BIC

def calc_r2(y, y_mod):
    '''
    Fucntion to calculate the Bayesian information criterion
    
    Input:
    ------
    y (np.array)                observed data
    y_mod (np.array)            fitted data

    Output:
    -------
    R2 (float)                  R squared

    '''   
    assert len(y) == len(y_mod)
    y = y.values.flatten()
    y_mod = y_mod.values.flatten()
    y[np.isnan(y_mod)] = np.nan
    y_mod[np.isnan(y)] = np.nan
    y = y[~np.isnan(y)]
    y_mod = y_mod[~np.isnan(y_mod)]
    n = len(y)
    MSE = np.mean(((y - y_mod)**2))
    R2 = 1 - MSE/np.mean((y - y.mean())**2)
    if (R2 > 1) | (R2 < 0): R2 = np.nan
    return R2

## reference: https://gdal.org/tutorials/geotransforms_tut.html
def trans_tif_grid(filename, center=True):
    '''
    Function to transform tiff grid to georeferenced latitude and longitude

    Input:
    ------
    filename (str)      name of the tiff file
    
    Output:
    ------
    lat (np.array)      latitude
    lon (np.array)      longitude

    Options:
    ------
    center (bool)       whether or not to use the center coordiante
                        default = True
    
    '''
    try:
        from osgeo import gdal
    except:
        raise ImportError("'gdal' libraries must be installed")
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

## concert from one code to another code
def convert_iso(regs, 
        region_from='ISO-Alpha3', 
        region_to='ISO-Numeric'):
    '''
    Function to convert between different region codes

    Input:
    ------
    regs (list)         list of regional ISO code
                        
    Output:
    ------
    regs_new (list)     new list of regional ISO code
    
    Options:
    --------
    region_from (str)   name of regional aggregation
                        default = 'ISO-Alpha3'
    region_to (str)     name of regional aggregation
                        default = 'ISO-Numeric'
    '''
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

    ## region mapping files to be loaded
    list_load = [zou for zou in os.listdir('./input_data/regions/') if all([_  in zou for _ in ['dico_', '.csv']])]

    ## load and create combined dictionary
    dico = {}
    for zou in list_load:
        if 'ISO' in zou: 
            with open('./input_data/regions/' + zou) as f: TMP = np.array([line for line in csv.reader(f)])
            if region_from in TMP[0,:].tolist() and region_to in TMP[0,:].tolist():
                if region_from == 'ISO-Numeric':
                    dico = {**dico, **{int(key):str(val) for key, val in zip(TMP[1:,TMP[0,:].tolist().index(region_from)], TMP[1:,TMP[0,:].tolist().index(region_to)])}}
                else:
                    dico = {**dico, **{str(key):str(val) for key, val in zip(TMP[1:,TMP[0,:].tolist().index(region_from)], TMP[1:,TMP[0,:].tolist().index(region_to)])}}

    regs_new = []
    regs_missing = []
    for reg in regs:
        try:
            if region_to == 'ISO-Numeric':
                regs_new.append(int(dico[reg]))
            else:
                regs_new.append(dico[reg])
        except KeyError:
            regs_missing.append(reg)
            regs_new.append(np.nan)
    
    if len(regs_missing) > 0: print("Code \033[1;30;42m{0}\033[0m doesn't exist.".format(regs_missing))
    return regs_new

## convert crop specifier to cropland type
def convert_crop_land(specifier, irr):
    '''
    Function to convert ISIMIP crop specifier to cropland type

    Input:
    ------
    specifier (str)         crop specifier
    irr (str)               irrigation
                            default = 'noirr'
    
    Output:
    ------
    var (list)               name of cropland type
    
    '''

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
##  FIND AND LOAD DATA
##################################################
def find_files_isimip3b_var(var, dir, 
        gcms=None, 
        scens=None, 
        keys=None):
    '''
    Find ISIMIP3b-related files and return name list

    Input:
    ------
    var (str)               variable (e.g., 'tas', 'pr')
    dir (str)               path to ISIMIP3b data

    Options:
    --------
    gcms (list)             Earth System Models (e.g., 'CanESM2' or 'CanESM5')
                            default = None
    scens (list)            a list of scenarios (e.g., ['ssp126' or 'ssp585'])
                            default = None
    keys (list)             a list of strings that must be included in the file names
                            default = None

    Output:
    -------
    files_out (list)        list of paths to all the filenames found for the given input

    '''
    class NotFound(Exception):
        pass

    assert (isinstance(gcms, list) if gcms is not None else True)
    assert (isinstance(scens, list) if scens is not None else True)
    assert (isinstance(keys, list) if keys is not None else True)

    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

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
def convert_doy(days_in, year, format='dd'):
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
    
    '''

    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

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
def trans_doy_gs(day_p, day_m, year, 
        method = 'forward', 
        time_scale = 'day'):
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
    '''

    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

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
        if is_leap(year):
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
## aggregate grid data to national or other regions based on given shapefile
## references: https://docs.xarray.dev/en/stable/user-guide/computation.html#weighted-array-reductions
def aggreg_grid(ds_in, 
        mask=None, 
        weight=None, 
        use_oscar=False, 
        use_isimip=False,
        shapefile='/mnt/d/Data/Shapefile/oscar/country.shp', 
        calc_global=True, 
        weight_scheme='area', 
        old_axis=['lat', 'lon'], 
        new_axis='reg_land', 
        time_axis='year', 
        weight_output=True, 
        weight_var='weight', 
        method='mean'):
    '''
    Function to aggregate data onto OSCAR regions. It uses dictionnaries mapping ISO regions to OSCAR regions defined in 'input_data/regions' by user.
    
    Input:
    ------
    ds_in (xr.Dataset)                  input dataset to be aggregated
        
    Output:
    -------
    ds_out (xr.Dataset)                 output dataset

    Options:
    --------
    mask (xr.Dataset | xr.DataArray)    input mask array
                                        default = None
    use_oscar (bool)                    whether to use standardized OSCAR mask file
                                        default = False
    use_isimip (bool)                   whether to use ISIMIP country mask file
                                        default = False
    shapefile (str)                     shapefile name
                                        default = '/mnt/d/Data/Shapefile/oscar/country.shp'
    calc_global (bool)                  whether to calculate the global average
                                        default = True
    weight_scheme (str)                 scheme of applying weight
                                        this is necessary for intensive variables (e.g. temperature that needs to be weighted by area); 
                                        default = 'area'
    weight (xr.DataArray)               additional weight variable; 
                                        default = None
    old_axis (list)                     name of regional axis that will be aggregated (must a dim of ds_in);
                                        default = ['lat', 'lon']
    new_axis (str)                      name of new aggregated regional axis (must NOT be in ds_in, and will be in ds_out);
                                        default = 'reg_land'
    time_axis (str)                     name of time axis (to ensure it is first dim in ds_out);
                                        default = 'year'
    weight_output (bool)                whether to output weight of the aggregated mask
                                        default = True
    weight_var (str)                    name of the output weight variable
                                        default = 'weight'
    method (str)                        calculation method of data, valid methods include 'mean' and 'sum'
                                        default = 'mean'
    
    '''
    warnings.filterwarnings('ignore')

    if use_isimip and use_oscar:
        print('\033[1;30;42m Conflicting mask setting! \033[0m')
        raise(RuntimeError)
    
    ## check old axis in ds_in and new_axis not in ds_in
    assert all(axis in ds_in.coords for axis in old_axis) and new_axis not in ds_in.coords
    if weight is not None: assert isinstance(weight, xr.DataArray)
    
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')
    t0 = time.time()

    ## make deep copy to be safe
    ds = ds_in.copy(deep=True)

    ## extract variables without regional axis
    ds_non = ds.drop([var for var in ds if (all(axis in ds[var].dims for axis in old_axis))] + old_axis)
    ds_out = ds.drop([var for var in ds if (any(axis not in ds[var].dims for axis in old_axis))])

    ## create mask
    if use_oscar:
        if mask is None:
            print("\033[1;30;42m Please load OSCAR mask file before running this procedure ...\033[0m")
            raise(RuntimeError)
        else:
            assert isinstance(mask, xr.Dataset)
            mask = sort_coords(mask, ds_in, axis=old_axis)
            mask = mask.rename({'reg_mask':new_axis, 'reg_mask_code':new_axis+'_code', 'reg_mask_name':new_axis+'_name'})
            mask_frac = mask['frac_reg']
            reg_numeric = mask[new_axis].values
    elif use_isimip:
        if mask is None:
            print("\033[1;30;42m Please load ISIMIP mask file before running this procedure...\033[0m")
        else:
            assert isinstance(mask, xr.Dataset)
            mask_frac = sort_coords(mask, ds_in, axis=old_axis)
            vars = np.array([var.replace('m_','') for var in mask_frac.data_vars])
            reg_numeric = np.array(convert_iso(vars))
            ## remove NaN values
            vars = vars[~np.isnan(reg_numeric)]
            reg_numeric = reg_numeric[~np.isnan(reg_numeric)].astype('int')
            vars = vars[np.argsort(reg_numeric)]
            reg_numeric = reg_numeric[np.argsort(reg_numeric)]
    else:
        if mask is None:
            print("\033[1;30;42m Better to create mask file in advance.\033[0m")
            print("Using shapefile: "+shapefile)
            mask = create_mask(shapefile, ds.lat, ds.lon)
        else:
            assert isinstance(mask, xr.DataArray)
        reg_numeric = np.sort(np.array(list(set(mask.values[np.nonzero(mask.values)]))))
    
    ## new regional aggregation
    ds_keep = ds_out
    
    ## create empty dataset with desired regional aggregation
    ds_out = ds_out.drop_dims(old_axis)
    ds_out = ds_out.drop_vars(var for var in ds_out)
    ds_out.coords[new_axis] = reg_numeric

    for var in ds_keep.data_vars:
        if all([axis in ds_keep[var].coords for axis in old_axis]):
            var_shape = ds_out[new_axis].shape
            var_coords = {ds_out[new_axis].name:ds_out[new_axis]}
            for coord in ds_keep[var].coords:
                if coord not in old_axis:
                    var_shape = var_shape+ds_keep[var][coord].shape
                    var_coords[coord] = ds_keep[var][coord]
            ds_out[var] = xr.DataArray(np.nan * np.zeros(var_shape), coords=var_coords)

    ## generate zero weight variable
    if weight_output:
        if weight is None:
            ds_out[weight_var] = xr.zeros_like(ds_out.coords[new_axis])
        else:
            ds_out[weight_var] = sum([np.nan * xr.zeros_like(ds_out.coords[dim]) for dim in  [new_axis] + [dim for dim in weight.dims if dim not in old_axis] ])

    if weight_scheme == 'area':
        dlat = np.abs(ds_in[old_axis[0]][0].values - ds_in[old_axis[0]][1].values)
        dlon = np.abs(ds_in[old_axis[1]][0].values - ds_in[old_axis[1]][1].values)
        w = calc_earth_cellarea(ds_in[old_axis[0]].values, dlat, dlon)
        w = xr.DataArray(w, dims=('lat',))
    elif weight_scheme == 'one': w = xr.ones_like(ds_in.lat)

    ## global average
    if calc_global:
        for var in ds_keep.data_vars:
            if method == 'mean': var_weighted = ds_keep[var].weighted(w).mean(old_axis)
            elif method == 'sum': var_weighted = ds_keep[var].weighted(w).sum(old_axis)
            ds_out[var+'_g'] = var_weighted
            try:
                ds_out[var+'_g'] = ds_out[var+'_g'].assign_attrs(units=ds_in[var].attrs['units'])
                ds_out[var+'_g'] = ds_out[var+'_g'].assign_attrs(long_name=ds_in[var].attrs['long_name']+' (global)')
            except KeyError:
                pass

    for reg in reg_numeric:
        if use_oscar:
            weight_reg = w*mask_frac.loc[{new_axis:reg}]
        elif use_isimip:
            var = 'm_'+vars[np.where(reg_numeric==reg, True, False)][0]
            weight_reg = w*mask_frac[var]
        else:
            weight_reg, mask = xr.broadcast(w, mask)
            weight_reg = xr.where(mask==reg, weight_reg, 0)
        ## additional weight
        if weight is not None: weight_reg = weight_reg*weight
        weight_reg = weight_reg.fillna(0)
        if weight_output: ds_out[weight_var].loc[{new_axis:reg}] = weight_reg.sum(old_axis)

        for var in ds_keep.data_vars:
            if method == 'mean': var_weighted = ds_keep[var].weighted(weight_reg).mean(old_axis, keep_attrs=True)
            elif method == 'sum': var_weighted = ds_keep[var].weighted(weight_reg).sum(old_axis, keep_attrs=True)
            ds_out[var].loc[{new_axis:reg}] = var_weighted
            try:
                ds_out[var] = ds_out[var].assign_attrs(long_name=ds_in[var].attrs['long_name']+' (regional)')
            except KeyError:
                pass

    ## merge with extracted variables
    ds_out = xr.merge([ds_out, ds_non])

    ## output weight
    if weight_output:
        ds_out['weight'] = ds_out['weight'].assign_attrs({'weight_scheme': weight_scheme, 'method': method})
        if method == 'area':
            ds_out['weight'] = ds_out['weight'].assign_attrs({'units': 'km^2'})

    ## add mask information for OSCAR regional aggregation
    ds_out = ds_out.assign(reg_land_code=mask['reg_land_code'], reg_land_name=mask['reg_land_name'])

    # make sure time axis is first
    if time_axis in ds_out.coords: 
        ds_out = ds_out.transpose(time_axis,...)
        for var in ds_out:
            if time_axis in ds_out[var].coords:
                ds_out[var] = ds_out[var].transpose(time_axis,...)

    t = time.time()-t0
    print("Total time consumed for aggregation : {:.0f}min {:.0f}sec".format(t // 60, t % 60))
    return ds_out

## aggregate sub-regional data to OSCAR regions
def aggreg_subreg(ds_in, var_list, 
        weight_var='weight', 
        reg_code='reg_land_code', 
        reg_list=['AUS', 'BRA', 'CAN', 'CHN', 'RUS', 'USA'],
        keep_sub=False):
    '''
    Function to create a global map of a given variable
    
    Input:
    ------
    ds_in (xr.Dataset)          input dataset with sub-regional aggregation
    var_list (list)             list of variables to be aggregated

    Output:
    -------
    ds_out (xr.Dataset)         output dataset with aggregated sub-regions

    Options:
    --------
    weight_var (str)            weight variable name
                                default = 'weight'
    reg_code (str)              axis name based on which to conduct aggregation
                                default = 'reg_land_code'
    reg_list (list)             list of regions to be aggregated
                                default = ['AUS', 'BRA', 'CAN', 'CHN', 'RUS', 'USA']
    keep_sub (bool)             whether to keep sub-regional values in the dataset
                                default = False

    '''
    warnings.filterwarnings('ignore')

    assert isinstance(var_list, list)
    assert isinstance(ds_in, xr.Dataset)
    assert all([weight_var in ds_in.data_vars, reg_code in ds_in.coords])
    assert all([var in ds_in.data_vars for var in var_list])

    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')

    ## swap the coordinate
    ds_in = ds_in.set_coords(reg_code)
    ds_in = ds_in.swap_dims({'reg_land': reg_code})

    ds_out = ds_in.copy(True)

    for key in reg_list:
        sub_regs = [reg for reg in ds_in.coords[reg_code].values if key+'-' in reg]
        for var in var_list:
            if reg_code in ds_out[var].coords:
                ds_out[var].loc[{reg_code: key}] = ds_in[var].loc[{reg_code: sub_regs}].weighted(ds_in[weight_var].loc[{reg_code: sub_regs}].fillna(0)).mean(reg_code).values
                if not keep_sub:
                    ## set all the subregional values to be NaN
                    ds_out[var].loc[{reg_code: sub_regs}] = np.nan * ds_in[var].loc[{reg_code: sub_regs}].values
                ds_out[weight_var].loc[{reg_code: key}] = ds_in[weight_var].loc[{reg_code: sub_regs}].sum(reg_code).values
                if not keep_sub:
                    ds_out[weight_var].loc[{reg_code: sub_regs}] = np.nan * ds_in[weight_var].loc[{reg_code: sub_regs}].values

    ## change to the original coordinate
    ds_out = ds_out.swap_dims({reg_code: 'reg_land'})

    return ds_out

## create mask
def create_mask(lat, lon, 
        shapefile='/mnt/d/Data/Shapefile/oscar/country.shp',
        save=False, 
        maskfile='mask.nc', 
        use_globe=True):
    '''
    Function to create the mask file of a given shapefile
    
    Input:
    ------
    lat (xr.DataArray)  input array of latitude
    lon (xr.DataArray)  input array of longitude
    
    Output:
    ------
    mask (xr.DataArray) output mask dataset
    
    Options:
    ------
    shapefile (str)     input shapefile name
                        default = '/mnt/d/Data/Shapefile/oscar/country.shp'
    save (bool)         whether or not to save mask file
                        default = True
    maskfile (str)      name of mask file
                        default = 'mask.nc'
    use_globe (bool)    whether or not to use 'global_land_mask' library
                        it is recommended to use this library when the land grid resolution is high
                        defualt = True
    '''
    warnings.filterwarnings('ignore')

    try:
        import geopandas
        from shapely.geometry import Point
    except:
        raise ImportError("'geopandas', 'shapely' and 'global_land_mask' libraries must be installed")
    
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')
    t0 = time.time()

    shp = geopandas.GeoDataFrame.from_file(shapefile)
    
    ## deal with longitude larger than 180
    lon = xr.where(lon > 180, lon-360, lon)
    points = np.array(np.meshgrid(lon, lat))
    
    ## create mask
    print("\033[1;30;42mStart creating mask.\033[0m")
    mask_data = np.zeros((len(lat), len(lon)), dtype=int)
    
    indice_keep = 0
    for i in np.arange(points.shape[1]):
        time_s = time.time()
        for j in np.arange(points.shape[2]):
            if use_globe:
                try:
                    from global_land_mask import globe
                except:
                    raise ImportError("'global_land_mask' library must be installed")
                if globe.is_land(points[:,i,j][1], points[:,i,j][0]):
                    pnt = Point(points[:,i,j])
                    for n in shp.length.keys():
                        indice = n + indice_keep
                        if indice >= shp.shape[0]:
                            indice = indice - shp.shape[0]
                        if shp.iloc[indice]['ISO_NUM'] and pnt.within(shp.iloc[indice]['geometry']):
                            mask_data[i,j] = int(shp.iloc[indice]['ISO_NUM'])
                            indice_keep = indice
                            break
            else:
                pnt = Point(points[:,i,j])
                for n in shp.length.keys():
                    indice = n + indice_keep
                    if indice >= shp.shape[0]:
                        indice = indice - shp.shape[0]
                    if shp.iloc[indice]['ISO_NUM'] and pnt.within(shp.iloc[indice]['geometry']):
                        mask_data[i,j] = int(shp.iloc[indice]['ISO_NUM'])
                        indice_keep = indice
                        break
                    
        time_e = time.time()
        time_r = time_e - time_s
        print("Latitude: ", lat[i].values)
        print("Longitudinal time consumed: {:.0f}min {:.0f}sec".format(time_r // 60, time_r % 60))
        
    print("Finished creating mask.")
    
    mask = xr.DataArray(data=mask_data, coords={'lat': lat, 'lon':lon}).astype('int')
    mask.name = 'ISO-Numeric'
    if save:
        mask.to_netcdf(maskfile)
    
    t = time.time()-t0
    print("Total time consumed fro creating mask: {:.0f}min {:.0f}sec".format(t // 60, t % 60))
    
    return mask

## extract regional data based on mask file
## https://docs.xarray.dev/en/stable/generated/xarray.DataArray.quantile.html
def extract_reg_stats(ds_in,
        quantiles=[0.05, 0.1, 1/6, 1/3, 0.5, 2/3, 5/6, 0.9, 0.95], 
        method='linear', 
        mask=None,                      
        use_oscar=True, 
        use_isimip=False,
        shapefile='/mnt/d/Data/Shapefile/oscar/country.shp', 
        old_axis=['lat', 'lon'], 
        new_axis='reg_land',
        time_axis='year'):
    '''
    Function to extract a certain region from gridded global data based on a give mask.
    
    Input:
    ------
    ds_in (xr.Dataset)

    Output:
    ------
    ds_out (xr.Dataset)                 output dataset
    
    Options:
    --------
    quantiles (list)                    the quantiles of regional values
                                        default = [0.05, 0.1, 1/6, 1/3, 0.5, 2/3, 5/6, 0.9, 0.95]
    method (str)                        interpolation method to use
                                        default = 'linear'
    mask (xr.Dataset|xr.DataArray)      input mask data
                                        default = None                        
    use_oscar (bool)                    whether to use standardized OSCAR mask file
                                        default = False
    use_isimip (bool)                   whether to use ISIMIP country mask file
                                        default = False
    shapefile (str)                     shapefile name
                                        default = '/mnt/d/Data/Shapefile/oscar/country.shp'
    old_axis (list)                     name of regional axis that will be aggregated (must a dim of ds_in);
                                        default = ['lat', 'lon']
    new_axis (str)                      name of new aggregated regional axis (must NOT be in ds_in, and will be in ds_out);
                                        default = 'reg_land'
    time_axis (str)                     name of time axis (to ensure it is first dim in ds_out);
                                        default = 'year'
    '''
    ## check old axis in ds_in and new_axis not in ds_in
    assert all(axis in ds_in.coords for axis in old_axis) and new_axis not in ds_in.coords

    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')
    t0 = time.time()

    ## make deep copy to be safe
    ds = ds_in.copy(deep=True)

    ## extract variables without regional axis
    ds_non = ds.drop([var for var in ds if (all(axis in ds[var].dims for axis in old_axis))] + old_axis)
    ds_out = ds.drop([var for var in ds if (any(axis not in ds[var].dims for axis in old_axis))])

    ## create mask
    if use_oscar:
        if mask is None:
            print("\033[1;30;42m Please load OSCAR mask file before running this procedure ...\033[0m")
            raise(RuntimeError)
        else:
            assert isinstance(mask, xr.Dataset)
            mask = mask.rename({'reg_mask':'reg_land', 'reg_mask_code':'reg_land_code', 'reg_mask_name':'reg_land_name'})
            mask = sort_coords(mask, ds_in, axis=old_axis)
            mask_frac = mask['frac_reg']
            reg_numeric = mask[new_axis].values
    elif use_isimip:
        if mask is None:
            print("\033[1;30;42m Please load ISIMIP mask file before running this procedure...\033[0m")
        else:
            assert isinstance(mask, xr.Dataset)
            mask_frac = sort_coords(mask, ds_in, axis=old_axis)
            vars = np.array([var.replace('m_','') for var in mask_frac.data_vars])
            reg_numeric = np.array(convert_iso(vars))
            ## remove NaN values
            vars = vars[~np.isnan(reg_numeric)]
            reg_numeric = reg_numeric[~np.isnan(reg_numeric)].astype('int')
            vars = vars[np.argsort(reg_numeric)]
            reg_numeric = reg_numeric[np.argsort(reg_numeric)]
    else:
        if mask is None:
            print("\033[1;30;42m Better to create mask file in advance.\033[0m")
            print("Using shapefile: "+shapefile)
            mask = create_mask(shapefile, ds.lat, ds.lon)
        else:
            assert isinstance(mask, xr.DataArray)
        reg_numeric = np.sort(np.array(list(set(mask.values[np.nonzero(mask.values)]))))

    ## new regional aggregation
    ds_keep = ds_out
    
    ## create empty dataset with desired regional aggregation
    ds_out = ds_out.drop_dims(old_axis)
    ds_out = ds_out.drop_vars(var for var in ds_out)
    ds_out.coords[new_axis] = reg_numeric

    for var in ds_keep.data_vars:
        var_shape = ds_out[new_axis].shape
        var_coords = {ds_out[new_axis].name:ds_out[new_axis]}
        for coord in ds_keep[var].coords:
            if coord not in old_axis:
                var_shape = var_shape+ds_keep[var][coord].shape
                var_coords[coord] = ds_keep[var][coord]
        for q in quantiles:
            ds_out['{:s}|{:.1f}th Percentile'.format(var, q*100)] = xr.DataArray(np.nan * np.zeros(var_shape), coords=var_coords)

    for reg in reg_numeric:
        mask_reg = mask_frac.sel(reg_land=reg, drop=True)
        for var in ds_keep.data_vars:
            for q in quantiles:
                mask_reg = mask_reg.broadcast_like(ds_in[var])
                var_sub = ds_in[var].where(mask_reg > 0)
                ds_out['{:s}|{:.1f}th Percentile'.format(var, q*100)].loc[{new_axis:reg}] = var_sub.quantile(q, dim=old_axis, method=method, keep_attrs=True, skipna=True)
                try:
                    ds_out['{:s}|{:.1f}th Percentile'.format(var, q*100)] = ds_out['{:s}|{:.1f}th Percentile'.format(var, q*100)].assign_attrs(long_name=ds_in[var].attrs['long_name']+' (regional)')
                except KeyError:
                    pass

    ## merge with extracted variables
    ds_out = xr.merge([ds_out, ds_non])

    ## add mask information for OSCAR regional aggregation
    ds_out = ds_out.assign(reg_land_code=mask['reg_land_code'], reg_land_name=mask['reg_land_name'])

    # make sure time axis is first
    if time_axis in ds_out.coords: 
        ds_out = ds_out.transpose(time_axis,...)
        for var in ds_out:
            if time_axis in ds_out[var].coords:
                ds_out[var] = ds_out[var].transpose(time_axis,...)

    t = time.time()-t0
    print("Total time consumed for aggregation : {:.0f}min {:.0f}sec".format(t // 60, t % 60))
    return ds_out

## split region into disaggreated regions
def split_region(reg, region_from, 
        region_to='ISO-Numeric'):
    '''
    Function to convert between different region codes
    Input:
    ------
    reg (str)           region to be disaggregated
    region_from (str)   name of regional aggregation (must a valid option)
                        
    Output:
    ------
    reg_new (list)      new list of regional code
    
    Options:
    --------
    region_to (str)     name of regional aggregation
                        default = 'ISO-Numeric'
    '''
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  for {reg} <<<')
    
    ## region mapping files to be loaded
    list_load = [zou for zou in os.listdir('./input_data/regions/') if all([_  in zou for _ in ['dico_', '.csv']])]

    ## load and create combined dictionary
    dico = {}
    for zou in list_load:
        if 'ISO' in zou: 
            with open('./input_data/regions/' + zou) as f: TMP = np.array([line for line in csv.reader(f)])
            if region_from in TMP[0,:].tolist() and region_to in TMP[0,:].tolist():
                for key, val in zip(TMP[1:,TMP[0,:].tolist().index(region_from)], TMP[1:,TMP[0,:].tolist().index(region_to)]):
                    if reg == key :
                        try:
                            dico.setdefault(reg, []).append(int(val))
                        except ValueError:
                            dico.setdefault(reg, []).append(str(val))
    reg_new = sorted(dico[reg])
        
    return reg_new

##################################################
##  REGIONAL PLOT
##################################################
## create global map for regional data
def create_global_map(var_in, 
        maskfile=None,
        axis='reg_land',
        levels=np.linspace(0,10,6)):
    '''
    Function to create a global map of a given variable
    
    Input:
    ------
    var_in (xr.DataArray)   1-D array
        
    Output:
    -------

    Options:
    --------
    maskfile (str)          maskfile
                            default = None
    axis (str)              regional axis
                            default = 'reg_land'
    levels (np.array)       levels of contour map
                            default = np.linspace(0, 10, 6)
    '''
    function_name = sys._getframe(  ).f_code.co_name
    print(f'>>>  Running {function_name}  <<<')
    warnings.filterwarnings('ignore')

    ## check old axis in ds_in and new_axis not in ds_in
    assert axis in var_in.coords
    
    if maskfile is not None:
        mask = xr.load_dataarray(maskfile)
    elif mask is None:
        raise RuntimeError('input maskfile is necessary')
    
    mask.name = 'reg_land'
    var = xr.DataArray(np.zeros_like(mask), dims=('lat','lon'), coords={'lat':mask.lat, 'lon':mask.lon})
    for reg in var_in.reg_land:
        try:
            var = xr.where(mask==reg, var_in.loc[{'reg_land':reg}], var)
        except KeyError:
            continue
        
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        print("============================")
        print("\033[1;30;42mPloting contour map ...\033[0m")
        crs = ccrs.PlateCarree(central_longitude=0.0)
        fig = plt.figure(figsize=(8, 5))
        ax = plt.subplot(111, projection=crs)
        ax.add_feature(cfeature.BORDERS, linewidth=0.75)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.75)
        # ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#97B6E1')
        ax.add_feature(ocean)
        cf = ax.contourf(var.lon, var.lat, var, levels, extend='max', cmap=plt.cm.RdYlBu_r, transform=ccrs.PlateCarree())
        cb = fig.colorbar(cf, orientation='horizontal', aspect=30, shrink=0.8, pad=0.08, extendrect='True')
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(-180, 210, 30), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 120, 30), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
    except ImportError:
        print("'cartopy' libraries must be installed")
    
    return ax