##################################################
##################################################

"""
CONTENT
-------
1. Nitrogen
    load_nitrogen_hist
    load_nitrogen_scen
2. Cropland area
    load_cropland_hist
    load_cropland_scen
3. Harvested area
    load_harvested_hist    
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
def load_Nfertl_hist(datasets=['ISIMIP3b-5crops', 'LUH2-v2h', 'Jagermeyr_2021', 'Adalibieke_2023'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    mod_region='sub-national',
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
                                default = ['ISIMIP3b-5crops', 'Jagermeyr_2021', 'Adalibieke_2023']
    crop_species (list)         crop species to be included
    '''
    
    ## crop type
    crop_type = {'c3ann':['ri1', 'ri2', 'ric', 'swh', 'wwh', 'whe'], 'c3nfx':['soy'], 'c4ann':['mai'], 'c3per': ['euc', 'pop'], 'c4per': ['mis']}
        
    ## main loading loop
    For0 = []
    units = {}

    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        for data in datasets:
            if os.path.isfile(f'input_data/drivers/crop/nitrogen_fertl__{data}_{mod_region}.nc'):
                data_dict[data] = f'{data}_{mod_region}' 
    if mod_region == 'national': 
        for data in datasets:
            for region in ['national', 'sub-national']:
                if os.path.isfile(f'input_data/drivers/crop/nitrogen_fertl__{data}_{region}.nc'):
                    data_dict[data] = f'{data}_{region}'
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading nitrogen_fertl__{name} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/nitrogen_fertl__{name}.nc')

        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
        
        ## sub-national dataset
        if data in ['Jagermeyr_2021']:
            For1 = xr.concat([For1.sel(soc='histsoc', drop=True), For1.sel(soc='2015soc', drop=True).isel(year=0).assign_coords(year=2015)], dim='year')
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old  != spc:
                    For = For1['N_fertl'].loc[{'spc_crop':spc_old}].assign_coords(spc_crop=spc).expand_dims('spc_crop', -1)
                    For1 = For1.combine_first(For)
            For1 = For1.drop_sel(spc_crop = [spc for spc in For1.spc_crop.values if spc not in crop_species])
        
        ## sub-national dataset
        if data in ['Adalibieke_2023']:
            For1['N_fertl'] = xr.DataArray(np.nansum([For1[var] for var in For1.data_vars], axis=0), dims=('year', 'spc_crop', 'reg_land'))
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old  != spc:
                    For = For1['N_fertl'].loc[{'spc_crop':spc_old}].assign_coords(spc_crop=spc).expand_dims('spc_crop', -1)
                    For1 = For1.combine_first(For)
            For1 = For1['N_fertl']
            For1 = For1.drop_sel(spc_crop = [spc for spc in For1.spc_crop.values if spc not in crop_species])
        
        ## sub-national dataset
        if data in ['ISIMIP3b-5crops']:
            For1 = xr.concat([For1.sel(soc='histsoc', drop=True), For1.sel(soc='2015soc', drop=True).isel(year=0).assign_coords(year=2015)], dim='year')
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl']
        
        ## sub-national dataset
        ## three scenarios: historical, historical-high, historical-low
        if data in ['LUH2-v2h']:
            For1 = For1.sel(year=slice(1850, 2015))
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl'].rename({'scen': 'data'})

        ## append to final list (with new dimension)
        if data in ['LUH2-v2h']:
            For0.append(For1.assign_coords(data=['LUH2-'+val for val in For1.coords['data'].astype(str).to_numpy()]))
        else:
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

    ## return
    return For


## historical nitrogen deposition
def load_Ndep_hist(datasets=['ISIMIP3b'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    mod_region='sub-national',
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
                                default = ['ISIMIP3b']
    crop_species (list)         crop species to be included
    '''
     
    ## main loading loop
    For0 = []
    units = {}
    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        for data in datasets:
            if os.path.isfile(f'input_data/drivers/crop/nitrogen_dep__{data}_{mod_region}.nc'):
                data_dict[data] = f'{data}_{mod_region}' 
    if mod_region == 'national': 
        for data in datasets:
            for region in ['national', 'sub-national']:
                if os.path.isfile(f'input_data/drivers/crop/nitrogen_dep__{data}_{region}.nc'):
                    data_dict[data] = f'{data}_{region}'
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading nitrogen_fertl__{name} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/nitrogen_dep__{name}.nc')
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
               
        if data in ['ISIMIP3b']:
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
def load_Nfertl_scen(datasets=['Jagermeyr_2021', 'ISIMIP3b-5crops'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    mod_region='sub-national',
    **useless):
    '''
    Function to load and format nitrogen input datasets from different scenarios, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['Jagermeyr_2021', 'ISIMIP3b-5crops']
    crop_species (list)         crop species to be included
    '''
    ## crop type
    crop_type = {'c3ann':['ri1', 'ri2', 'ric', 'swh', 'wwh', 'whe'], 'c3nfx':['soy'], 'c4ann':['mai']}
        
    ## main loading loop
    For0 = []
    units = {}
    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        for data in datasets:
            if os.path.isfile(f'input_data/drivers/crop/nitrogen_fertl__{data}_{mod_region}.nc'):
                data_dict[data] = f'{data}_{mod_region}' 
    if mod_region == 'national': 
        for data in datasets:
            for region in ['national', 'sub-national']:
                if os.path.isfile(f'input_data/drivers/crop/nitrogen_fertl__{data}_{region}.nc'):
                    data_dict[data] = f'{data}_{region}'
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading nitrogen_fertl__{name} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/nitrogen_fertl__{name}.nc')
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
        
        if data in ['Jagermeyr_2021']:
            For1 = For1.sel(soc='2015soc').isel(year=0, drop=True)
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old  != spc:
                    For = For1['N_fertl'].loc[{'spc_crop':spc_old}].assign_coords(spc_crop=spc).expand_dims('spc_crop', -1).rename({'soc': 'scen'})
                    For1 = For1.combine_first(For)
            For1 = For1.drop_sel(spc_crop = [spc for spc in For1.spc_crop.values if spc not in crop_species])
            For1['scen'] = data+'-'+For1['scen']
                           
        if data in ['ISIMIP3b-5crops']:
            For1 = For1.drop_sel(soc='histsoc')
            for spc in crop_species:
                for typ, spcs in crop_type.items():
                    if spc in spcs:
                        For = For1['fertl_'+typ].expand_dims('spc_crop', -1).assign_coords(spc_crop=[spc]).rename('N_fertl')
                        For1 = For1.combine_first(For.to_dataset())
            For1 = For1['N_fertl'].to_dataset().rename({'soc':'scen'})
            For1['scen'] = data+'-'+For1['scen']          
    
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
def load_Ndep_scen(datasets=['ISIMIP3b'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    mod_region='sub-national',
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
                                default = ['ISIMIP3b']
    crop_species (list)         crop species to be included
    '''
        
    ## main loading loop
    For0 = []
    units = {}
    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        for data in datasets:
            if os.path.isfile(f'input_data/drivers/crop/nitrogen_dep__{data}_{mod_region}.nc'):
                data_dict[data] = f'{data}_{mod_region}' 
    if mod_region == 'national': 
        for data in datasets:
            for region in ['national', 'sub-national']:
                if os.path.isfile(f'input_data/drivers/crop/nitrogen_dep__{data}_{region}.nc'):
                    data_dict[data] = f'{data}_{region}'
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading nitrogen_fertl__{name} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/nitrogen_dep__{name}.nc')
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
                                  
        if data in ['ISIMIP3b']:
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
def load_harvested_hist(datasets=['Adalibieke_2023', 'Waha_2020', 'Becker-Reshef_2023', 'CROPGRIDSv1.07', 'RiceAtlas', 'MIRCA2000', 'Monfreda_2008', 'ric-ISIMIP3b', 'FAO'],
    crop_species=['mai', 'ri1', 'ri2', 'ri3', 'soy', 'swh', 'wwh'],
    mod_region='sub-national',
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
                                default = ['Waha_2020', 'Becker-Reshef_2023', 'RiceAtlas', 'FAO']
    crop_species (list)         crop species to be included
    '''
    from core_fct.fct_pre_CROP import aggreg_subreg       
    ## main loading loop
    For0 = []
    units = {}

    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        for data in datasets:
            if os.path.isfile(f'input_data/drivers/crop/harvested-area__{data}_{mod_region}.nc'):
                data_dict[data] = f'{data}_{mod_region}' 
    if mod_region == 'national': 
        for data in datasets:
            for region in ['national', 'sub-national']:
                if os.path.isfile(f'input_data/drivers/crop/harvested-area__{data}_{region}.nc'):
                    data_dict[data] = f'{data}_{region}'
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading harvested-area__{name} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/harvested-area__{name}.nc')

        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
           
        ## sub-national dataset
        if data in ['Adalibieke_2023', 'Waha_2020', 'Becker-Reshef_2023', 'MIRCA2000', 'Monfreda_2008', 'ric-ISIMIP3b']:
            if data in ['Waha_2020', 'ric-ISIMIP3b']:
                if 'ric' in crop_species: For1 = For1.combine_first(For1['Ah'].sel(spc_crop=['ri1', 'ri2', 'ri3']).sum('spc_crop').assign_coords(spc_crop='ric'))
            if data in ['Becker-Reshef_2023']:
                if 'whe' in crop_species: For1 = For1.combine_first(For1['Ah'].sel(spc_crop=['swh', 'wwh']).sum('spc_crop').assign_coords(spc_crop='whe'))
            if data == 'ric-ISIMIP3b': For1 = For1.squeeze()
            if data == 'MIRCA2000': For1 = For1.sum(dim='irr')
            if mod_region == 'national': For1 = aggreg_subreg(For1['Ah'])

        ## national dataset
        if data in ['CROPGRIDSv1.07', 'RiceAtlas', 'FAO']:
            if data in ['RiceAtlas']: 
                if 'ric' in crop_species: For1 = For1.combine_first(For1['Ah'].sel(spc_crop=['ri1', 'ri2', 'ri3']).sum('spc_crop').assign_coords(spc_crop='ric'))
            if mod_region == 'sub-national': 
                continue
            else:
                For1 = For1['Ah']

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
    For = For.drop_sel(spc_crop=[spc for spc in For.spc_crop.values if spc not in crop_species])
    For = For.transpose(*(['year', 'spc_crop', 'reg_land'] + [var for var in For.coords if 'data' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For

## historical crop-specific cropland area
def load_cropland_hist(datasets=['ISIMIP3b-15crops', 'Jackson_2019', 'CROPGRIDSv1.07'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    irr_flag=False,
    mod_region='sub-national',
    **useless):
    '''
    Function to load and format hitorical cropland area datasets, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['ISIMIP3b']
    crop_species (list)         crop species to be included
                                default = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    irr_flag (bool)             whether to include irirgation information or not
                                default = False
    '''
           
    ## main loading loop
    from core_fct.fct_pre_CROP import convert_crop_land, aggreg_subreg
    For0 = []
    units = {}

    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        if irr_flag:
            data_dict['ISIMIP3b-15crops'] = 'ISIMIP3b-15crops_sub-national'
        else:
            for data in datasets:
                if os.path.isfile(f'input_data/drivers/crop/cropland__{data}_{mod_region}.nc'):
                    data_dict[data] = f'{data}_{mod_region}' 
    elif mod_region == 'national': 
        if irr_flag:
            data_dict['ISIMIP3b-15crops'] = 'ISIMIP3b-15crops_sub-national'
        else:
            for data in datasets:
                for region in ['national', 'sub-national']:
                    if os.path.isfile(f'input_data/drivers/crop/cropland__{data}_{region}.nc'):
                        data_dict[data] = f'{data}_{region}'
    ## check missing datasets
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading cropland__{name} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/cropland__{name}.nc')
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))

        ## sub-national dataset
        ## with irrigation information
        if data == 'ISIMIP3b-15crops':
            For1 = For1.sel(soc='histsoc', drop=True)
            area_spc_list = []
            for spc in crop_species:
                area_irr_list = []
                for irr in ['firr', 'noirr']:
                    try:
                        area_spc = sum([For1[luc_spc] for luc_spc in convert_crop_land(spc, irr)])
                        area_irr_list.append(area_spc.assign_coords(irr=irr).expand_dims('irr'))
                    except KeyError:
                        raise RuntimeWarning('Key error: {0} under {1} is not in {2})'.format(spc, irr, data))
                area_irr = xr.concat(area_irr_list, dim='irr')
                if irr_flag: 
                    area_spc_list.append(area_irr.assign_coords(spc_crop=spc).expand_dims('spc_crop'))
                else:
                    area_spc_list.append(area_irr.sum('irr').assign_coords(spc_crop=spc).expand_dims('spc_crop'))
            For1 = xr.concat(area_spc_list, dim='spc_crop').rename('Ac')
            if mod_region == 'national': For1 = aggreg_subreg(For1)

        if data in ['Jackson_2019', 'CROPGRIDSv1.07']:
            area_spc_list = []
            for spc in crop_species:
                spc_old = spc
                if spc in ['ri1', 'ri2']: spc_old = 'ric'
                if spc in ['swh', 'wwh']: spc_old = 'whe'
                if spc_old != spc: 
                    area_spc_list.append(For1['Ac'].sel(spc_crop=spc_old).assign_coords(spc_crop=spc))
                else:
                    area_spc_list.append(For1['Ac'].sel(spc_crop=spc_old))
            For1 = xr.concat(area_spc_list, dim='spc_crop')
            if mod_region == 'national' and data in ['Jackson_2019']: 
                For1 = aggreg_subreg(For1)

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
    if irr_flag:
        For = For.transpose(*(['year', 'spc_crop', 'irr', 'reg_land'] + [var for var in For.coords if 'data' in var]))
    else:
        For = For.transpose(*(['year', 'spc_crop', 'reg_land'] + [var for var in For.coords if 'data' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For

## crop-specific cropland area scenarios
def load_cropland_scen(datasets=['ISIMIP3b-15crops'],
    crop_species=['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh'],
    irr_flag=False,
    mod_region='sub-national',
    **useless):
    '''
    Function to load and format cropland area under different scenarios, taken from the 'input_data' folder.
    
    Input:
    ------

    Output:
    -------
    For (xr.Dataset)            dataset that contains the loaded datasets

    Options:
    --------
    datasets (list)             names of primary datasets to be loaded;
                                default = ['ISIMIP3b-15crops']
    crop_species (list)         crop species to be included
                                default = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    irr_flag (bool)             whether to include irirgation information or not
                                default = False                           
    '''
        
    ## main loading loop
    from core_fct.fct_pre_CROP import convert_crop_land, aggreg_subreg
    For0 = []
    units = {}

    data_dict = {}
    ## check regional level
    if mod_region == 'sub-national':
        for data in datasets:
            if os.path.isfile(f'input_data/drivers/crop/cropland__{data}_{mod_region}.nc'):
                data_dict[data] = f'{data}_{mod_region}' 
    if mod_region == 'national': 
        for data in datasets:
            for region in ['national', 'sub-national']:
                if os.path.isfile(f'input_data/drivers/crop/cropland__{data}_{region}.nc'):
                    data_dict[data] = f'{data}_{region}'
    datasets_missing = [data for data in datasets if data not in data_dict.keys()]
    if len(datasets_missing) > 0:
        print(f'The following datasets are not compatible: {datasets_missing}')

    print(f'Loading {len(data_dict)} datasets for {mod_region} region')
    for data, name in data_dict.items():
        print(f'Loading cropland__{data} ...')
        For1 = xr.load_dataset(f'input_data/drivers/crop/cropland__{data}.nc')
        
        ## get and check units
        for VAR in For1:
            if 'units' in For1[VAR].attrs:
                if VAR not in units.keys():
                    units[VAR] = For1[VAR].units
                else:
                    if units[VAR] != For1[VAR].units:
                        raise RuntimeWarning('inconsistent units: {0} (internal dic) vs. {1} ({2} in {3})'.format(units[VAR], For1[VAR].units, "'"+VAR+"'", "'"+data+"'"))
    
        ## sub-national dataset
        ## with irrigation information
        if data == 'ISIMIP3b-15crops':
            For1 = For1.drop_sel(soc='histsoc')
            scens = [scen for scen in For1.soc.values]
            area_scen_list = []
            for scen in scens:
                for_scen = For1.sel(soc=scen).squeeze()
                if scen in ['1850soc', '2015soc']: for_scen = For1.isel(year=0, drop=True).squeeze()
                area_spc_list = []
                for spc in crop_species:
                    area_irr_list = []
                    for irr in ['firr', 'noirr']:
                        try:
                            area_spc = sum([for_scen[luc_spc] for luc_spc in convert_crop_land(spc, irr)])
                            area_irr_list.append(area_spc.assign_coords(irr=irr).expand_dims('irr'))
                        except KeyError:
                            raise RuntimeWarning('Key error: {0} under {1} is not in {2} scenario {3})'.format(spc, irr, data, scen))
                        area_irr = xr.concat(area_irr_list, dim='irr')
                    if irr_flag:
                        area_spc_list.append(area_irr.assign_coords(spc_crop=spc).expand_dims('spc_crop'))
                    else:
                        area_spc_list.append(area_irr.sum('irr').assign_coords(spc_crop=spc).expand_dims('spc_crop'))
                area_scen_list.append(xr.concat(area_spc_list, dim='spc_crop').assign_coords(scen=data+'-'+scen).expand_dims('scen'))
            For1 = xr.concat(area_scen_list, dim='scen').rename('Ac')
            if mod_region == 'national': For1 = aggreg_subreg(For1)

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
    if irr_flag:
        For = For.transpose(*([var for var in For.coords if var == 'year'] + ['reg_land','spc_crop', 'irr'] + [var for var in For.coords if 'scen' in var]))
    else:
        For = For.transpose(*([var for var in For.coords if var == 'year'] + ['reg_land', 'spc_crop'] + [var for var in For.coords if 'scen' in var]))

    ## reapply units
    for VAR in For:
        if VAR in units.keys():
            For[VAR].attrs['units'] = units[VAR]
            
    ## deal with fillvalue
    For = For.fillna(0)

    ## return
    return For
