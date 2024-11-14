##################################################
##################################################

"""
CONTENT
-------
1. Nitrogen
    load_nitrogen_hist
    load_nitrogen_scen
2. Cropland area
    load_area_hist    
Z. WRAPPERS
    load_all_hist
    load_all_scen
"""

##################################################
##################################################

import os
import warnings
import numpy as np
import xarray as xr

##################################################
##   1. NITROGEN
##################################################

## historical nitrogen fertilizer input
def load_Nfertl_hist(datasets=['Jagermeyr_2021', 'Adalibieke_2023', 'ISIMIP2a', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b', 'LUH2-v2h'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    **useless):
    '''
    Function to load and format hitorical nitrogen input datasets, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['Jagermeyr_2021', 'Adalibieke_2023', 'ISIMIP2a', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']
    crop_species (list)         crop species to be included
    '''
    
    ## crop type
    crop_type = {'c3ann':['ri1', 'ri2', 'ric', 'swh', 'wwh', 'whe'], 'c3nfx':['soy'], 'c4ann':['mai']}
        
    ## main loading loop
    For0 = []
    units = {}
    for data in datasets:

        ## load data if available
        if os.path.isfile('input_data/drivers/crop/nitrogen_fertl__' + data + '.nc'):
            For1 = xr.load_dataset('input_data/drivers/crop/nitrogen_fertl__' + data + '.nc')

        ## display message otherwise
        else: raise IOError('{0} not available'.format(data))
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
        
        ## (Jagermeyr et al., 2021; doi:10.1038/s43016-021-00400-y)
        if data in ['Jagermeyr_2021']:
            For1 = For1.sel(soc='histsoc', drop=True)
            vars = [var for var in For1.data_vars if var != 'N_manure']
            for var in vars: For1[var] = For1[var]+For1['N_manure']
            For1 = For1.drop_vars(['N_manure'])
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old  != spc:
                    For = For1['N_fertl'].loc[{'spc_crop':spc_old}].assign_coords(spc_crop=spc).expand_dims('spc_crop', -1)
                    For1 = For1.combine_first(For)
            For1 = For1.drop_sel(spc_crop = [spc for spc in For1.spc_crop.values if spc not in crop_species])
        
        ## (Adalibieke et al., 2023; doi:10.1038/s41597-023-02526-z)
        if data in ['Adalibieke_2023']:
            For1['N_fertl'] = xr.DataArray(np.nansum([For1[var] for var in For1.data_vars], axis=0), dims=('year', 'spc_crop', 'reg_land'))
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old  != spc:
                    For = For1['N_fertl'].loc[{'spc_crop':spc_old}].assign_coords(spc_crop=spc).expand_dims('spc_crop', -1)
                    For1 = For1.combine_first(For)
            For1 = For1['N_fertl'].to_dataset()
            For1 = For1.drop_sel(spc_crop = [spc for spc in For1.spc_crop.values if spc not in crop_species])
        
        if data in ['ISIMIP2a', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']:
            For1 = For1.sel(soc='histsoc', drop=True)
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl'].to_dataset()
        
        ## three scenarios: historical, high, low
        if data in ['LUH2-v2h']:
            For1 = For1.sel(scen='historical', drop=True)
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl'].to_dataset()
        
        ## append to final list (with new dimension)
        For0.append(For1.expand_dims('data', -1).assign_coords(data=[data]))
        del For1

    ## merge into one xarray
    For0 = xr.merge(For0)
            
    ## create one data axis per driver
    For = xr.Dataset()
    for VAR in For0:
        TMP = [For0[VAR].sel(data=data).rename({'data':'data_'+VAR}) for data in For0.data.values if not np.isnan(For0[VAR].sel(data=data).sum(min_count=1))]
        For[VAR] = xr.concat(TMP, dim='data_'+VAR)
        del TMP
    
    ## order dimensions
    For = For.transpose(*(['year', 'spc_crop', 'reg_land'] + [var for var in For.coords if 'data' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For


## historical nitrogen deposition
def load_Ndep_hist(datasets=['ISIMIP2a', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    **useless):
    '''
    Function to load and format hitorical nitrogen input datasets, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['ISIMIP2a', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']
    crop_species (list)         crop species to be included
    '''
     
    ## main loading loop
    For0 = []
    units = {}
    for data in datasets:

        ## load data if available
        if os.path.isfile('input_data/drivers/crop/nitrogen_dep__' + data + '.nc'):
            For1 = xr.load_dataset('input_data/drivers/crop/nitrogen_dep__' + data + '.nc')

        ## display message otherwise
        else: raise IOError('{0} not available'.format(data))
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
               
        if data in ['ISIMIP2a', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']:
            For1 = For1.sel(soc='histsoc', drop=True)
            For1['N_dep'] = For1['nhx'].fillna(0)+For1['noy'].fillna(0)
            For1 = For1['N_dep'].to_dataset()
        
        ## append to final list (with new dimension)
        For0.append(For1.expand_dims('data', -1).assign_coords(data=[data]))
        del For1

    ## merge into one xarray
    For0 = xr.merge(For0)
            
    ## create one data axis per driver
    For = xr.Dataset()
    for VAR in For0:
        TMP = [For0[VAR].sel(data=data).rename({'data':'data_'+VAR}) for data in For0.data.values if not np.isnan(For0[VAR].sel(data=data).sum(min_count=1))]
        For[VAR] = xr.concat(TMP, dim='data_'+VAR)
        del TMP
    
    ## order dimensions
    For = For.transpose(*(['year', 'reg_land'] + [var for var in For.coords if 'data' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For


## scenarios for nitrogen fertilizer input
def load_Nfertl_scen(datasets=['Jagermeyr_2021', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b', 'LUH2-v2f'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    **useless):
    '''
    Function to load and format hitorical nitrogen input datasets, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['Jagermeyr_2021', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']
    crop_species (list)         crop species to be included
    '''
    ## crop type
    crop_type = {'c3ann':['ri1', 'ri2', 'ric', 'swh', 'wwh', 'whe'], 'c3nfx':['soy'], 'c4ann':['mai']}
        
    ## main loading loop
    For0 = []
    units = {}
    for data in datasets:

        ## load data if available
        if os.path.isfile('input_data/drivers/crop/nitrogen_fertl__' + data + '.nc'):
            For1 = xr.load_dataset('input_data/drivers/crop/nitrogen_fertl__' + data + '.nc')

        ## display message otherwise
        else: raise IOError('{0} not available'.format(data))
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
        
        if data in ['Jagermeyr_2021']:
            For1 = For1.drop_sel(soc='histsoc')
            vars = [var for var in For1.data_vars if var != 'N_manure']
            for var in vars: For1[var] = For1[var]+For1['N_manure']
            For1 = For1.drop_vars(['N_manure'])
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old  != spc:
                    For = For1['N_fertl'].loc[{'spc_crop':spc_old}].assign_coords(spc_crop=spc).expand_dims('spc_crop', -1)
                    For1 = For1.combine_first(For)
            For1 = For1.drop_sel(spc_crop = [spc for spc in For1.spc_crop.values if spc not in crop_species]).rename({'soc':'scen'})
            For1['scen'] = data+'-'+For1['scen']
                           
        if data in ['ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']:
            For1 = For1.drop_sel(soc='histsoc')
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl'].to_dataset().rename({'soc':'scen'})
            For1['scen'] = data+'-'+For1['scen']          
    
        ## three scenarios
        if data in ['LUH2-v2f']:
            For1 = For1.stack(new_scen=('climate', 'model'))
            For1 = For1.reset_index('new_scen')
            For1['new_scen'] = [var1+'_'+var2 for var1, var2 in zip(For1.climate.values, For1.model.values)]
            For1 = For1.dropna('new_scen', 'all').rename({'new_scen':'scen'})
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl'].to_dataset()
            
        ## append to final list (with new dimension)
        For0.append(For1)
        del For1

    ## merge into one xarray
    For0 = xr.merge(For0)
            
    ## create one data axis per driver
    For = xr.Dataset()
    for VAR in For0:
        TMP = [For0[VAR].sel(scen=scen).rename({'scen':'scen_'+VAR}) for scen in For0.scen.values if not np.isnan(For0[VAR].sel(scen=scen).sum(min_count=1))]
        For[VAR] = xr.concat(TMP, dim='scen_'+VAR)
        del TMP
    
    ## order dimensions
    For = For.transpose(*(['year', 'spc_crop', 'reg_land'] + [var for var in For.coords if 'scen' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For

## scenarios for nitrogen deposition
def load_Ndep_scen(datasets=['ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    **useless):
    '''
    Function to load and format hitorical nitrogen input datasets, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['Jagermeyr_2021', 'ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']
    crop_species (list)         crop species to be included
    '''
        
    ## main loading loop
    For0 = []
    units = {}
    for data in datasets:

        ## load data if available
        if os.path.isfile('input_data/drivers/crop/nitrogen_dep__' + data + '.nc'):
            For1 = xr.load_dataset('input_data/drivers/crop/nitrogen_dep__' + data + '.nc')

        ## display message otherwise
        else: raise IOError('{0} not available'.format(data))
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
                                  
        if data in ['ISIMIP2b', 'ISIMIP3a', 'ISIMIP3b']:
            For1 = For1.drop_sel(soc='histsoc')
            For1['N_dep'] = For1['nhx'].fillna(0)+For1['noy'].fillna(0)
            For1 = For1['N_dep'].rename({'soc':'scen'})
            For1['scen'] = data+'-'+For1['scen']          
    
        ## append to final list (with new dimension)
        For0.append(For1)
        del For1

    ## merge into one xarray
    For0 = xr.merge(For0)
            
    ## create one data axis per driver
    For = xr.Dataset()
    for VAR in For0:
        TMP = [For0[VAR].sel(scen=scen).rename({'scen':'scen_'+VAR}) for scen in For0.scen.values if not np.isnan(For0[VAR].sel(scen=scen).sum(min_count=1))]
        For[VAR] = xr.concat(TMP, dim='scen_'+VAR)
        del TMP
    
    ## order dimensions
    For = For.transpose(*(['year', 'reg_land'] + [var for var in For.coords if 'scen' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For

##################################################
##   2. Cropland area
##################################################

## historical harvested area
def load_area_hist(datasets=['Adalibieke_2023', 'FAO'],
    crop_species=['mai', 'ric', 'soy', 'whe'],
    **useless):
    '''
    Function to load and format hitorical nitrogen input datasets, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['Adalibieke_2023', 'FAO']
    crop_species (list)         crop species to be included
    '''
           
    ## main loading loop
    For0 = []
    units = {}
    for data in datasets:

        ## load data if available
        if os.path.isfile('input_data/drivers/area-harvested__' + data + '.nc'):
            For1 = xr.load_dataset('input_data/drivers/area-harvested__' + data + '.nc')

        ## display message otherwise
        else: raise IOError('{0} not available'.format(data))
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
            
        ## append to final list (with new dimension)
        For0.append(For1.expand_dims('data', -1).assign_coords(data=[data]))
        del For1

    ## merge into one xarray
    For0 = xr.merge(For0)
            
    ## create one data axis per driver
    For = xr.Dataset()
    for VAR in For0:
        TMP = [For0[VAR].sel(data=data).rename({'data':'data_'+VAR}) for data in For0.data.values if not np.isnan(For0[VAR].sel(data=data).sum(min_count=1))]
        For[VAR] = xr.concat(TMP, dim='data_'+VAR)
        del TMP
    
    ## order dimensions
    For = For.transpose(*(['year', 'spc_crop', 'reg_land'] + [var for var in For.coords if 'data' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For
