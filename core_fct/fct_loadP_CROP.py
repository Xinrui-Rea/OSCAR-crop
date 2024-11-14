##################################################
##################################################

"""
CONTENT
-------
1. YIELD
    1.1. CO2
        load_CO2_misc
        load_RC_obs
    1.2. CLIMATE
        load_climate_ISIMIP3a
    1.3. FERTILIZER
        load_Nfertl_ISIMIP3b
        load_Ndep_ISIMIP3b
        load_Nbnf_misc
        load_RN_obs
    1.4. CROP
        load_YD_ISIMIP3b
Z. WRAPPER
    load_ISIMIP_param
"""

##################################################
##################################################
import os
import numpy as np
import xarray as xr

##################################################
## 1. YIELD
##################################################

##=================
## 1.1. CO2
##=================
def load_CO2_misc(**useless):
    ## initialization
    Par = xr.Dataset()
    
    ## preindustrail CO2 concentration defined by ISIMIP 
    Par['CO2_pi'] = xr.DataArray(284.73, attrs={'units':'ppm'})
    Par['CO2_2015'] = xr.DataArray(399.95, attrs={'units':'ppm'})

    ## return
    return Par

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
## 1.2. CLIMATE
##=================
## relationship between land temperaturen and land temperature
def load_Tl_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile('input_data/parameters/crop/Tl_ISIMIP3b__regional.nc') and not recalibrate:
        Par = xr.load_dataset('input_data/parameters/crop/Tl_ISIMIP3b__regional.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par

## relationship between growing season precipitation and land precipitation
def load_Pl_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile('input_data/parameters/crop/Pl_ISIMIP3b__regional.nc') and not recalibrate:
        Par = xr.load_dataset('input_data/parameters/crop/Pl_ISIMIP3b__regional.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par

## relationship between growing season temperature and land temperature
def load_Tgs_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile('input_data/parameters/crop/Tgs_ISIMIP3b__regional.nc') and not recalibrate:
        Par = xr.load_dataset('input_data/parameters/crop/Tgs_ISIMIP3b__regional.nc')
    ## otherwise, launch calibration
    else:
        raise RuntimeError('embedded calibration not available yet')
    ## return
    return Par

## relationship between growing season precipitation and land precipitation
def load_Pgs_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile('input_data/parameters/crop/Pgs_ISIMIP3b__regional.nc') and not recalibrate:
        Par = xr.load_dataset('input_data/parameters/crop/Pgs_ISIMIP3b__regional.nc')
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

##================
## 1.3. FERTILIZER
##================
## nitrogen fertilizer input under 2015 socio-economic scenario
def load_Nfertl_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile('input_data/parameters/crop/N_fertl_ISIMIP3b__regional.nc') and not recalibrate:
        Par =  xr.load_dataset('input_data/parameters/crop/N_fertl_ISIMIP3b__regional.nc')
    ## otherwise, launch calibration
    else:
        from core_fct.fct_calib_CROP import calib_Nfertl_ISIMIP3b
        Par = calib_Nfertl_ISIMIP3b()

    ## return
    return Par

## nitrogen deposition under 2015 socio-economic scenario
def load_Ndep_ISIMIP3b(recalibrate=False, **useless):
    ## load from existing file
    if os.path.isfile('input_data/parameters/crop/N_dep_ISIMIP3b__regional.nc') and not recalibrate:
        Par =  xr.load_dataset('input_data/parameters/crop/N_dep_ISIMIP3b__regional.nc')
    ## otherwise, launch calibration
    else:
        from core_fct.fct_calib_CROP import calib_Ndep_ISIMIP3b
        Par = calib_Ndep_ISIMIP3b()

    ## return
    return Par


## biological nitrogen fixation
def load_Nbnf_misc(**useless):
    ## initialization
    Par = xr.load_dataset('/h/u145/liuxinrui/CROP/input_data/regions/region_coords.nc')
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    Par.coords['mod_bnf_soy'] = ['Ma_2022', 'Peoples_2009']

    from core_fct.fct_pre_CROP import split_region
    with_subregs = ['AUS', 'BRA', 'CAN', 'CHN', 'RUS', 'USA']
    Par = Par.set_coords('reg_land_code')
    Par = Par.swap_dims({'reg_land': 'reg_land_code'})

    ## non-symbiotic BNF, and based on:
    ## (Ladha et al., 2022; doi:10.1016/j.fcr.2022.108541) (Table 11)
    Par['N_bnf'] = xr.DataArray(np.tile(np.array([12.7, 22.4, 22.4, np.nan, 12.7, 12.7]), (len(Par.reg_land_code), len(Par.mod_bnf_soy), 1)).transpose(2, 0, 1),
        dims=('spc_crop', 'reg_land_code', 'mod_bnf_soy'), attrs={'units':'kgN ha-1'})
    
    ## continental-level BNF of soybean, and based on:
    ## (Ma et al., 2022; doi:10.5194/gmd-15-815-2022) (Table 3)
    Par['N_bnf'].loc[{'spc_crop':'soy'}] = 132

    ## BNF in South Asia
    regions = split_region('Southern Asia', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Ma_2022'}] = 53
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Peoples_2009'}] = 88
    
    ## BNF in Southeast Asia
    regions = split_region('South-eastern Asia', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Ma_2022'}] = 141
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Peoples_2009'}] = 115
        
    ## BNF in Africa
    regions = split_region('Africa', region_from='Region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Ma_2022'}] = 172
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Peoples_2009'}] = 193
    
    ## BNF in North America
    regions = split_region('Northern America', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Ma_2022'}] = 127
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Peoples_2009'}] = 144
    
    ## BNF in South America
    regions = split_region('Latin America and the Caribbean', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Ma_2022'}] = 156
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions), 'mod_bnf_soy':'Peoples_2009'}] = 136
    
    ## BNF in East Asia
    regions = split_region('Eastern Asia', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions)}] = 101
    
    ## BNF in Central Asia
    regions = split_region('Central Asia', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions)}] = 63
    
    ## BNF in West Asia
    regions = split_region('Western Asia', region_from='Sub-region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions)}] = 27
    
    ## BNF in Europe
    regions = split_region('Europe', region_from='Region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions)}] = 117
    
    ## BNF in Oceania
    regions = split_region('Oceania', region_from='Region Name', region_to='ISO-Alpha3')
    regions = [reg for reg in regions if reg in Par.reg_land_code.values]
    regs = list(set(regions).intersection(set(with_subregs)))
    for reg in regs:
        regions = regions+split_region(reg, region_from='Code', region_to='Sub-code')
    Par['N_bnf'].loc[{'spc_crop':'soy', 'reg_land_code':np.array(regions)}] = 78
    
    Par = Par.swap_dims({'reg_land_code': 'reg_land'})
    return Par


def load_RN_obs(**useless):
    ## initialization
    Par = xr.load_dataset('./input_data/regions/region_coords.nc')
    Par.coords['spc_crop'] = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    Par.coords['mod_RN_fct'] = ['2nd', 'MM', 'Mit', 'Geo']
    
    ## option to choose response functional form
    Par['RN_is_2nd'] = xr.DataArray([True, False, False, False], dims='mod_RN_fct')
    Par['RN_is_MM'] = xr.DataArray([False, True, False, False], dims='mod_RN_fct')
    Par['RN_is_Mit'] = xr.DataArray([False, False, True, False], dims='mod_RN_fct')
    Par['RN_is_Geo'] = xr.DataArray([False, False, False, True], dims='mod_RN_fct')
    
    ## regression coefficient of response function between nitrogen input and crop yield, and based on:
    ## (van Grinsven et al., 2022; doi:10.1038/s43016-021-00447-x)
    ## parameters of 2nd polynomial
    ## global parameter (supplementary note 5)
    Par['g_a'] = xr.DataArray(-1.870e-05*np.ones((len(Par.spc_crop), len(Par.reg_land)), dtype=float),
                dims=('spc_crop', 'reg_land'), attrs={'units':'(kgN ha-1)-2'})
    Par['g_b'] = xr.DataArray(8.768e-03*np.ones((len(Par.spc_crop), len(Par.reg_land)), dtype=float),
                dims=('spc_crop', 'reg_land'), attrs={'units':'(kgN ha-1)-1'})
    
    Par = Par.swap_dims({'reg_land': 'reg_land_code'})
    ## UK, winter wheat (supplementary note 3)
    ## used as parameters for both spring wheat and winter wheat
    Par['g_a'].loc[{'spc_crop': 'swh', 'reg_land_code': 'GBR'}] = -1.345e-05
    Par['g_b'].loc[{'spc_crop': 'swh', 'reg_land_code': 'GBR'}] = 7.291e-03
    
    ## USA, maize (supplementary note 4)
    subregs = ['USA-AK', 'USA-HI', 'USA-XA', 'USA-XD', 'USA-XH', 'USA-XC', 'USA-XF', 'USA-XE', 'USA-XG', 'USA-XB', 'USA-XI']
    Par['g_a'].loc[{'spc_crop':'mai', 'reg_land_code': subregs + ['USA']}] = -1.758e-05
    Par['g_b'].loc[{'spc_crop':'mai', 'reg_land_code': subregs + ['USA']}] = 8.379e-03

    Par = Par.swap_dims({'reg_land_code': 'reg_land'})

    ## South Asia, rice + wheat (supplementary note 8)
    ## used as global parameters for response of rice yield
    Par['g_a'].loc[{'spc_crop': ['ri1', 'ri2']}] = -4.396e-05
    Par['g_b'].loc[{'spc_crop': ['ri1', 'ri2']}] = 4.261e-03

    ## parameters of Michaelis-Menten function
    Par['g2_a'] = xr.DataArray(1.352*np.ones(len(Par.spc_crop), dtype=float),
                    dims=('spc_crop', ), attrs={'units':'1'})
    Par['g2_b'] = xr.DataArray(87*np.ones(len(Par.spc_crop), dtype=float),
                    dims=('spc_crop', ), attrs={'units':'kgN ha-1'})

    ## parameters of Mitcherlich function
    Par['g3_a'] = xr.DataArray(-1.045*np.ones(len(Par.spc_crop), dtype=float),
                    dims=('spc_crop',), attrs={'units':'1'})
    Par['g3_b'] = xr.DataArray(-0.012*np.ones(len(Par.spc_crop), dtype=float),
                    dims=('spc_crop',), attrs={'units':'kgN ha-1'})

    ## parameters of George function
    Par['g4_a'] = xr.DataArray(-1.290*np.ones(len(Par.spc_crop), dtype=float),
                    dims=('spc_crop',), attrs={'units':'1'})
    Par['g4_b'] = xr.DataArray(-8.109e-04*np.ones(len(Par.spc_crop), dtype=float),
                    dims=('spc_crop',), attrs={'units':'(kgN ha-1)-1'})
    
    ## return
    return Par

##================
## 1.5. YIELD
##================
## crop yield parameters calibrated on different models participating in ISIMIP3b
def load_YD_ISIMIP3b(models=['CYGMA1p74', 'EPIC-IIASA', 'ISAM', 'LDNDC', 'LPJmL','PEPIC', 'PROMET', 'SIMPLACE-LINTUL5'], **useless):
    Par = xr.Dataset()
    Par.coords['mod_YD_crop'] = models
    Par.coords['fct_YD'] = np.arange(1, 11)
    Par['RC_switch'] = xr.DataArray(np.zeros((len(models), len(Par.fct_YD))), dims=('mod_YD_crop', 'fct_YD'))
    Par['RT_switch'] = xr.DataArray(np.zeros((len(models), len(Par.fct_YD))), dims=('mod_YD_crop', 'fct_YD'))
    Par['RP_switch'] = xr.DataArray(np.zeros((len(models), len(Par.fct_YD))), dims=('mod_YD_crop', 'fct_YD'))
    
    Par0 = []
    ## load from existing file
    for model in models:
        if os.path.isfile('input_data/parameters/crop/crop_'+model+'__regional.nc'):
            Par1 = xr.load_dataset('input_data/parameters/crop/crop_'+model+'__regional.nc')
            Par['RC_switch'].loc[{'mod_YD_crop':model, 'fct_YD':Par1['fct_CO2'].values}] = 1
            Par['RT_switch'].loc[{'mod_YD_crop':model, 'fct_YD':Par1['fct_Tgs'].values}] = 1
            Par['RP_switch'].loc[{'mod_YD_crop':model, 'fct_YD':Par1['fct_Pgs'].values}] = 1
            Par1 = Par1.drop_vars(['fct_CO2', 'fct_Tgs', 'fct_Pgs'])
            Par0.append(Par1.expand_dims('mod_YD_crop', -1).assign_coords(mod_YD_crop=[model]))
        else:
            raise RuntimeError(model+' parameter dataset not found')
    
    Par0.append(Par)
    Par = xr.merge(Par0)
    
    ## return
    return Par


##################################################
##   Z. WRAPPER
##################################################

## wrapping function
def load_ISIMIP_param(mod_region='regional', recalibrate=False):
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
    load_list = [load_CO2_misc, 
        load_Tl_ISIMIP3b, load_Tgs_ISIMIP3b, 
        load_Pl_ISIMIP3b, load_Pgs_ISIMIP3b, 
        load_Nfertl_ISIMIP3b, load_Ndep_ISIMIP3b, load_Nbnf_misc, load_RN_obs, 
        load_YD_ISIMIP3b]
    
    ## return all
    return xr.merge([load(mod_region=mod_region, recalibrate=recalibrate) for load in load_list])
