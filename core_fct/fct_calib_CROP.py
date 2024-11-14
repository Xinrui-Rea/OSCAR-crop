import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

##################################################
## 1. YIELD
##################################################

##================
## 1.4. FERTILIZER
##================
def calib_Nfertl_ISIMIP3b(path_in='input_data/drivers/crop/', path_out='input_data/parameters/crop/', **useless):
    
    crop_type = {'c3ann':['ri1', 'ri2', 'ric', 'swh', 'wwh', 'whe'], 'c3nfx':['soy'], 'c4ann':['mai']}
    ## load original data
    da = xr.load_dataset(path_in+'nitrogen_fertl__ISIMIP3b.nc')
    da = da.sel(soc='2015soc', drop=True).isel(year=0, drop=True)
    
    Par = xr.Dataset()
    Par.coords['spc_crop'] = ['mai', 'soy', 'ri1', 'ri2', 'swh', 'wwh']
    Par.coords['reg_land'] = da.reg_land
    Par['N_fertl_0'] = xr.DataArray(np.zeros((len(Par.spc_crop), len(Par.reg_land)), dtype=float),
                        dims=('spc_crop', 'reg_land'))
    
    for spc in Par.spc_crop:
        for key, value in crop_type.items():
            if spc in value: Par['N_fertl_0'].loc[{'spc_crop':spc}] = da['fertl_'+key].fillna(0)
                
    if path_out is not None: Par.to_netcdf(path_out+'N_fertl_ISIMIP3b__regional.nc')
    
    return Par

def calib_Ndep_ISIMIP3b(path_in='input_data/drivers/crop/', path_out='input_data/parameters/crop/', **useless):
    
    ## load original data
    da = xr.load_dataset(path_in+'nitrogen_dep__ISIMIP3b.nc')
    da = da.sel(soc='2015soc', drop=True).isel(year=0, drop=True)
    
    Par = xr.Dataset()
    Par['N_dep_0'] = da['nhx'].fillna(0)+da['noy'].fillna(0)
                
    if path_out is not None: Par.to_netcdf(path_out+'N_dep_ISIMIP3b__regional.nc')
    
    return Par
    