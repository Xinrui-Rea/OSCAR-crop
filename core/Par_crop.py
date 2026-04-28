##################################################
##################################################

"""
CONTENT
-------
1. YIELD
    1.1. CO2
        load_CO2_misc
        load_RC_obs
    1.2. TEMPERATURE
        load_Tl_ISIMIP3b
        load_Tgs_ISIMIP3b
        load_RT_obs
    1.3. PRECIPITATION
        load_Pl_ISIMIP3b
        load_Pgs_ISIMIP3b
    1.4. FERTILIZER
        load_Ndep_ISIMIP3b
        load_Nbnf_misc
        load_RN_obs
    1.5. CROP
        load_YD_ISIMIP3b
2. ABOVEGROUND BIOMASS
    2.1 HARVEST INDEX
        load_HI_ISIMIP3b
Z. WRAPPER
    load_ISIMIP_param
    load_nitro_param
"""

##################################################
##################################################
import os
import numpy as np
import xarray as xr

from core.paths import path_data
##################################################
## 1. YIELD
##################################################

##=================
## 1.1. CO2
##=================
## CO2-yield response parameters based on observations
def load_RC_obs(**useless):
    ## initialization
    Par = xr.Dataset()
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    
    ## linear regression coefficient of response function between D_CO2 and crop yield
    Par['g_CO2'] = xr.DataArray([7.2731e-04, 3.5022e-04, 3.5022e-04, 1.0988e-03, 9.1892e-04, 9.1892e-04],
                        dims=('spc_crop', ), attrs={'units': 'ppm-1'})
    Par['g_CO2_unc'] = xr.DataArray([3.5466e-04, 1.3338e-04, 1.3338e-04, 1.3874e-04, 9.0380e-05, 9.0380e-05],
                        dims=('spc_crop', ), attrs={'units': 'ppm-1', 'range': '1std'})

    ## return
    return Par

##=================
## 1.2. TEMPERATURE
##=================
## relationship between land temperaturen and land temperature
def load_Tl_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/Tl__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par = xr.load_dataset(f'{path_data}parameters/Tl__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par


## relationship between growing season temperature and land temperature
def load_Tgs_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/Tgs__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par = xr.load_dataset(f'{path_data}parameters/Tgs__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par

## growing season temperature parameters based on observations
## TODO: reconsider the functional forms
def load_RT_obs(**useless):
    Par = xr.Dataset()
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    
    ## optimal growing season temperature
    Par['T_opt'] = xr.DataArray([294.28, 296.80, 296.80, 292.97, 285.64, 285.64], 
                                dims=('spc_crop', ), attrs={'units':'K'})
    Par['T_opt_unc'] = xr.DataArray([0.6178, 0.6729, 0.6729, 3.9962, 0.6648, 0.6648], 
                                dims=('spc_crop', ), attrs={'units':'K', 'range':'1std'})
    
    ## 2nc order coefficients
    Par['g_Tgs2'] = xr.DataArray([-0.006628, -0.013942, -0.013942, -0.001109, -0.007695, -0.007695], 
                                dims=('spc_crop', ), attrs={'units':'K-2'})
    Par['g_Tgs2_unc'] = xr.DataArray([-0.006628, -0.013942, -0.013942, -0.001109, -0.007695, -0.007695], 
                                dims=('spc_crop', ), attrs={'units':'K-2', 'range':'1std'})

    ## return
    return Par

##===================
## 1.3. PRECIPITATION
##===================
## relationship between growing season precipitation and land precipitation
def load_Pl_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/Pl__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par = xr.load_dataset(f'{path_data}parameters/Pl__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par

## relationship between growing season precipitation and land precipitation
def load_Pgs_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/Pgs__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par = xr.load_dataset(f'{path_data}parameters/Pgs__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par

##================
## 1.4. FERTILIZER
##================
## nitrogen deposition under 2015 socio-economic scenario
def load_Ndep_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/N_dep__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par =  xr.load_dataset(f'{path_data}parameters/N_dep__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        from core.calib_crop import calib_Ndep_ISIMIP3b
        Par = calib_Ndep_ISIMIP3b()
        print('Nitrogen deposition parameters calibrated and loaded.')

    ## return
    return Par

## biological nitrogen fixation
def load_Nbnf_misc(**useless):
    ## initialization
    from core.utils_crop import load_reg_coords
    Par = load_reg_coords(mod_region='Sub-national')
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    Par.coords['mod_bnf_soy'] = ['Ma_2022', 'Peoples_2009']

    from core.utils_crop import convert_reg_code
    ## non-symbiotic BNF, and based on:
    ## (Ladha et al., 2022; doi:10.1016/j.fcr.2022.108541) (Table 11)
    Par['N_bnf'] = xr.DataArray(np.tile(np.array([12.7, 22.4, 22.4, np.nan, 12.7, 12.7]), (len(Par.reg_code), len(Par.mod_bnf_soy), 1)).transpose(2, 0, 1),
        dims=('spc_crop', 'reg_code', 'mod_bnf_soy'), attrs={'units':'kgN ha-1'})
    
    ## continental-level BNF of soybean, and based on:
    ## (Ma et al., 2022; doi:10.5194/gmd-15-815-2022) (Table 3)
    Par['N_bnf'].loc[{'spc_crop':'soy'}] = 132

    ## BNF in South Asia
    reg_list = convert_reg_code('Southern Asia', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Ma_2022'}] = 53
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Peoples_2009'}] = 88

    ## BNF in Southeast Asia
    reg_list = convert_reg_code('South-eastern Asia', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Ma_2022'}] = 141
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Peoples_2009'}] = 115

    ## BNF in Africa
    reg_list = convert_reg_code('Africa', region_from='Continent', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Ma_2022'}] = 172
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Peoples_2009'}] = 193

    ## BNF in North America
    reg_list = convert_reg_code('Northern America', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Ma_2022'}] = 127
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Peoples_2009'}] = 144

    ## BNF in South America
    reg_list = convert_reg_code('Latin America and the Caribbean', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Ma_2022'}] = 156
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list, 'mod_bnf_soy':'Peoples_2009'}] = 136
    
    ## BNF in East Asia
    reg_list = convert_reg_code('Eastern Asia', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':np.array(reg_list)}] = 101

    ## BNF in Central Asia
    reg_list = convert_reg_code('Central Asia', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':np.array(reg_list)}] = 63
    
    ## BNF in West Asia
    reg_list = convert_reg_code('Western Asia', region_from='Sub-region', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list}] = 27
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list}] = 27
    
    ## BNF in Europe
    reg_list = convert_reg_code('Europe', region_from='Continent', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list}] = 117

    ## BNF in Oceania
    reg_list = convert_reg_code('Oceania', region_from='Continent', region_to='Sub-national')
    reg_list = [reg for reg in reg_list if reg in Par.reg_code.values]
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_code':reg_list}] = 78

    return Par

def load_RN_obs(use_mit=True, use_geo=False, use_MM=False, use_2nd=False, use_unc=True, **useless):
    ## initialization
    Par = xr.Dataset()
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    Par.coords['unc_Norm'] = ['mean', 'std']

    if sum([use_mit, use_geo, use_MM, use_2nd]) != 1:
        print('Error loading N-yield response parameters, must choose one from Mitscherlich, George, Michaelis-Menten, or 2nd polynomial functions.')
        raise RuntimeError
 
    ## use Mitscherlich function
    if use_mit:
        Par['g_a'] = xr.DataArray([[-1.0237, -1.6855, -1.6855, 0, -1.0519, -1.0519],
                                   [0.0355, 0.9069, 0.9069, 0, 0.0261, 0.0261]],
                                   dims=('unc_Norm', 'spc_crop'), 
                                   attrs={'units':'1', 'description':'Mitscherlich function'})
        Par['g_b'] = xr.DataArray([[-1.1053, -0.3543, -0.3543, 0, -1.2303, -1.2303],
                                   [0.1126, 0.2658, 0.2658, 0, 0.0860, 0.0860]],
                                   dims=('unc_Norm', 'spc_crop'), 
                                   attrs={'units':'1', 'description':'Mitscherlich function'})

    ## use George function
    if use_geo:
        Par['g_a'] = xr.DataArray([[-2997.80, -1546.2, -1546.2, 0, -4497.18, -4497.18],
                                   [177.16, 903.30, 903.30, 0, 202.54, 202.54]],
                                   dims=('unc_Norm', 'spc_crop'), 
                                   attrs={'units':'1', 'description':'George function'})
        Par['g_b'] = xr.DataArray([[-29.34, -14.96, -14.96, 0, -44.25, -44.25],
                                   [1.7568, 8.9923, 8.9923, 0, 2.0155, 2.0155]],
                                   dims=('unc_Norm', 'spc_crop'),
                                   attrs={'units':'1', 'description':'George function'})

    ## use 2nd function
    if use_2nd:
        Par['g_a'] = xr.DataArray([[-0.1488, -0.0772, -0.0772, 0, -0.2242, -0.2242],
                                   [0.0088, 0.0450, 0.0450, 0, 0.0101, 0.0101]],
                                   dims=('unc_Norm', 'spc_crop'), 
                                   attrs={'units':'1', 'description':'2nd polynomial function'})
        Par['g_b'] = xr.DataArray([[0.7835, 0.5830, 0.5830, 0, 0.9477, 0.9477],
                                   [0.0248, 0.0884, 0.0884, 0, 0.0208, 0.0208]],
                                   dims=('unc_Norm', 'spc_crop'), attrs={'units':'1', 'description':'2nd polynomial function'})

    ## use Michaelis-Menten function
    if use_MM:
        Par['g_a'] = xr.DataArray([[1.3085, 2.8975, 2.8975, 0, 1.3791, 1.3791],
                                   [0.0833, 1.9833, 1.9833, 0, 0.0605, 0.0605]],
                                   dims=('unc_Norm', 'spc_crop'), 
                                   attrs={'units':'1', 'description':'Michaelis-Menten function'})
        Par['g_b'] = xr.DataArray([[0.9298, 4.7743, 4.7743, 0, 0.8718, 0.8718],
                                   [0.1696, 4.4828, 4.4828, 0, 0.1034, 0.1034]],
                                   dims=('unc_Norm', 'spc_crop'), 
                                   attrs={'units':'1', 'description':'Michaelis-Menten function'})

    if not use_unc: Par = Par.isel(unc_Norm='Mean').drop_vars('unc_Norm').squeeze()

    return Par

##================
## 1.5 YIELD
##================
def load_YD_ISIMIP3b(models=['CYGMA1p74', 'EPIC-IIASA', 'ISAM', 'LDNDC', 'LPJmL','PEPIC', 'PROMET', 'SIMPLACE-LINTUL5'], recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/YD__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par = xr.load_dataset(f'{path_data}parameters/YD__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    
    for model in Par.mod_YD_crop.values:
        if model not in models: Par = Par.drop_sel(mod_YD_crop=model)

    ## return
    return Par

##################################################
##   2. ABOVEGROUND BIOMASS
##################################################
def load_HI_ISIMIP3b(models=['CYGMA1p74', 'EPIC-IIASA', 'ISAM', 'LDNDC', 'LPJmL','PEPIC', 'PROMET', 'SIMPLACE-LINTUL5'], recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile(f'{path_data}parameters/HI__ISIMIP3b_sub-national.nc') and not recalibrate:
        Par = xr.load_dataset(f'{path_data}parameters/HI__ISIMIP3b_sub-national.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    
    for model in Par.mod_YD_crop.values:
        if model not in models: Par = Par.drop_sel(mod_YD_crop=model)

    ## return
    return Par

def load_RY_obs(**useless):
    ## initialization
    Par = xr.Dataset()
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    
    ## carbon content in straw biomass
    ## https://doi.org/10.5194/essd-17-369-2025, Table 2
    Par['fc'] = xr.DataArray([0.55, 0.53, 0.53, 0.51, 0.51, 0.51],
                        dims=('spc_crop', ), attrs={'units': '1'})

    ## return
    return Par

##################################################
##   Z. WRAPPER
##################################################

## wrapping all ISIMIP3b-related function
def load_ISIMIP3b_param(mod_region='regional', recalibrate=False, **useless):
    '''
    Wrapper function to load all primary parameters.
    
    Input:
    ------
    mod_region (str)        regional aggregation name       

    Output:
    -------
    Par (xr.Dataset)        merged dataset

    Options:
    --------
    recalibrate (bool)      whether to recalibrate all possible parameters;
                            WARNING: currently not working;
                            default = False
    '''

    print('loading primary parameters')

    ## list of loading fuctions
    load_list = [
        load_Tl_ISIMIP3b, load_Tgs_ISIMIP3b, 
        load_Pl_ISIMIP3b, load_Pgs_ISIMIP3b, 
        load_YD_ISIMIP3b,
        load_Ndep_ISIMIP3b, load_Nbnf_misc, load_RN_obs,
        load_HI_ISIMIP3b
        ]
    
    ## return all
    return xr.merge([load(mod_region=mod_region, recalibrate=recalibrate) for load in load_list]).transpose('spc_crop', 'reg_code', 'irr', ...)

## wrapping all nitrogen response function parameters
def load_nitro_param(mod_region='regional', recalibrate=False, use_unc=True, **useless):
    '''
    Wrapper function to load all nitrogen parameters.

    Input:
    ------
    mod_region (str)        regional aggregation name
    recalibrate (bool)      whether to recalibrate all possible parameters;
                            WARNING: currently not working;
                            default = False

    Output:
    -------
    Par (xr.Dataset)        merged dataset
    '''

    print('loading nitrogen parameters')

    ## list of loading fuctions
    load_list = [
        load_Nbnf_misc, load_Ndep_ISIMIP3b, load_RN_obs
    ]

    ## return all
    return xr.merge([load(mod_region=mod_region, recalibrate=recalibrate, use_unc=use_unc) for load in load_list]).transpose('spc_crop', 'reg_code', ...)
