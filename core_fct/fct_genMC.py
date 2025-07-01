"""
Copyright: IIASA (International Institute for Applied Systems Analysis), 2016-2021; CEA (Commissariat a L'Energie Atomique) & UVSQ (Universite de Versailles et Saint-Quentin), 2016
Contributor(s): Thomas Gasser (gasser@iiasa.ac.at), Yann Quilcaille

This software is a computer program whose purpose is to simulate the behavior of the Earth system, with a specific but not exclusive focus on anthropogenic climate change.

This software is governed by the CeCILL license under French law and abiding by the rules of distribution of free software.  You can use, modify and/ or redistribute the software under the terms of the CeCILL license as circulated by CEA, CNRS and INRIA at the following URL "http://www.cecill.info". 

As a counterpart to the access to the source code and rights to copy, modify and redistribute granted by the license, users are provided only with a limited warranty and the software's author, the holder of the economic rights, and the successive licensors have only limited liability. 

In this respect, the user's attention is drawn to the risks associated with loading, using, modifying and/or developing or reproducing the software by the user in light of its specific status of free software, that may mean that it is complicated to manipulate, and that also therefore means that it is reserved for developers and experienced professionals having in-depth computer knowledge. Users are therefore encouraged to load and test the software's suitability as regards their requirements in conditions enabling the security of their systems and/or data to be ensured and,  more generally, to use and operate it in the same conditions as regards security. 

The fact that you are presently reading this means that you have had knowledge of the CeCILL license and that you accept its terms.
"""

##################################################
##################################################

import random
import warnings
import numpy as np
import xarray as xr
import scipy.stats as st

from scipy.integrate import quad
from scipy.optimize import fsolve


##################################################
##   1. ANCILLARY FUNCTIONS
##################################################

## function to get lognorm distrib parameters
def lognorm_distrib_param(mean, std):
    mu = np.log(mean / np.sqrt(1. + std**2./mean**2.))
    sigma = np.sqrt(np.log(1. + std**2./mean**2.))
    return mu, sigma


## function to infer logitnorm distrib parameters
def logitnorm_distrib_param(mean, std):
    ## error function
    def err(par):
        exp, _ = quad(lambda x, mu, sigma: 1/(1.-x) * 1./np.sqrt(2*np.pi*sigma**2.) * np.exp(-0.5*(np.log(x/(1.-x))-mu)**2./sigma**2.), 0, 1, args=tuple(par), limit=100)
        var, _ = quad(lambda x, mu, sigma: x/(1.-x) * 1./np.sqrt(2*np.pi*sigma**2.) * np.exp(-0.5*(np.log(x/(1.-x))-mu)**2./sigma**2.), 0, 1, args=tuple(par), limit=100)
        return np.array([exp-mean, np.sqrt(var-exp**2)-std])**2
    ## minimize error function
    try:
        par, _, fsolve_flag, _ = fsolve(err, [np.log(mean/(1.-mean)), np.sqrt(std/mean)], full_output=True)
        mu, sigma = par[0], np.abs(par[1])
    except ZeroDivisionError:
        fsolve_flag = 0
    ## return
    if fsolve_flag == 1: return mu, sigma
    else: return np.nan, np.nan

##################################################
## 2. DEAL WITH PARAMETERS
##################################################
def extrat_crop_par(Par0):
    '''
    Function to extract parameters for the crop emulator

    Input:
    ------
    Par0 (xr.Dataset)       dataset containing initial parameters

    Output:
    -------
    Par (xr.Dataset)        dataset with parameters from different functional forms extracted seperately

    Options:
    -------

    '''
    print(f'extracting crop emulator parameters')

    Par = Par0.copy(deep=True)
    var_list = [f'{chr(97+i)}_{var}' for i in range(3) for var in ['CO2', 'Tgs', 'Pgs']]
    for var in var_list:
        for num in np.arange(7):
            Par[f'{var}_{str(num)}'] = Par0[var].where(Par0[f'fct_{var.split('_')[-1]}'] == num)
    Par = Par.drop_vars(var_list)
    Par = Par.drop_vars(['fct_CO2', 'fct_Tgs', 'fct_Pgs'])
    return Par    

def restore_crop_par(Par0):
    '''
    Function to restore parameters for the crop emulator

    Input:
    ------
    Par0 (xr.Dataset)       dataset containing extrated parameters

    Output:
    -------
    Par (xr.Dataset)        dataset with parameters restored to original format

    Options:
    -------

    '''
    print(f'restoring crop emulator parameters')

    Par = Par0.copy(deep=True)
    var_list = ['CO2', 'Tgs', 'Pgs']
    for var in var_list:
        Par[f'fct_{var}'] = np.nan * Par[f'a_{var}_0']
        for i in range(3):
            Par[f'{chr(97+i)}_{var}'] = Par[f'{chr(97+i)}_{var}_0']
            for num in range(7):
                Par[f'fct_{var}'] = xr.where(Par0[f'{chr(97+i)}_{var}_{str(num)}'].notnull(), num, Par[f'fct_{var}'])
                Par[f'{chr(97+i)}_{var}'] = xr.where(Par0[f'{chr(97+i)}_{var}_{str(num)}'].notnull(), Par0[f'{chr(97+i)}_{var}_{str(num)}'], Par[f'{chr(97+i)}_{var}'])
    
    Par = Par.drop_vars([f'{chr(97+i)}_{var}_{str(j)}' for i in range(3) for j in range(7) for var in ['CO2', 'Tgs', 'Pgs']])
    return Par

##################################################
## 3. GENERATE MONTE CARLO PARAMETERS
##################################################

## generate all Monte Carlo configurations 
def generate_config(Par0, nMC, kde_to_mod=False, mod_to_unc=False, mod_noise=0.1, kde_bw=None, seed=None, par_exclude=None):
    '''
    Function to generate Monte Carlo configuration (= parameters) for OSCAR.
    
    Input:
    ------
    Par0 (xr.Dataset)       dataset containing initial parameters
    nMC (int)               number of MC elements
    
    Output:
    -------
    Par_mc (xr.Dataset)     dataset containing MC parameters

    Options:
    --------
    kde_to_mod (bool)       turn all kde_ options to mod_ options;
                            default = False
    mod_to_unc (bool)       turn all mod_ options to unc_ options;
                            default = False
    mod_noise (float)       equivalent s.d. of relative noise added on top of mod_ options;
                            default = 0.1
    kde_bw                  bandwith option for kde_ options forwarded to scipy.stats.gaussian_kde;
                            default = None
    seed (int)              seed for random number generation forwarded to numpyp.random.default_rng;
                            default = None
    par_exclude (list)      list of parameters to exclude from the MC generation;
                            default = None
    '''

    print('generating MC configurations')

    ## copy as precaution
    Par = Par0.copy(deep=True)

    ## list mod_ and kde_ dimensions
    mod_list = [coo for coo in Par.coords if coo[:4] == 'mod_']
    kde_list = [coo for coo in Par.coords if coo[:4] == 'kde_']

    ## list uncertainty options, parameters and check no mixing
    par_unc_list, par_mod_list, par_kde_list = [], [], []
    for par in Par:
        is_unc = any(['unc_' in dim for dim in Par[par].dims])
        is_mod = any(['mod_' in dim for dim in Par[par].dims])
        is_kde = any(['kde_' in dim for dim in Par[par].dims])
        if is_unc + is_mod + is_kde > 1:
            raise RuntimeError("Cannot mix unc_, mod_ and/or kde_ approaches; change parameter '{}'".format(par))     
        elif is_unc: 
            if par_exclude is None or par not in par_exclude:
                par_unc_list.append(par)
        elif is_mod: 
            if par_exclude is None or par not in par_exclude:
                par_mod_list.append(par)
        elif is_kde: 
            if par_exclude is None or par not in par_exclude:
                par_kde_list.append(par)

    ## turn kde_ into mod_ (if requested)
    if kde_to_mod:
        Par = Par.rename({kde: kde.replace('kde_', 'mod_', 1) for kde in kde_list})
        mod_list, kde_list = mod_list + kde_list, []
        par_mod_list, par_kde_list = par_mod_list + par_kde_list, []

    ## turn mod_ to unc_ (if requested)
    ## assumes functional form based on provided values
    if mod_to_unc:
        for par in par_mod_list:
            if (Par[par] == Par[par].mean()).all():
                Par[par] = xr.DataArray(Par[par].mean(), attrs=Par[par].attrs)
                par_mod_list.remove(par)
            elif (Par[par] == Par[par]**2).all(): # switch
                Par[par] = xr.DataArray([0, 1], coords=['mini', 'maxi'], dims='unc_Choice', attrs=Par[par].attrs)
            elif (Par[par] >= 0).all() and (Par[par] <= 1).all():
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_LogitNorm', attrs=Par[par].attrs)
            elif (Par[par] >= 0).all() or (Par[par] <= 0).all():
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_LogNorm', attrs=Par[par].attrs)
            else:
                Par[par] = xr.DataArray([Par[par].mean(), Par[par].std()], coords=['mean', 'std'], dims='unc_Norm', attrs=Par[par].attrs)
        mod_list = []
        par_unc_list, par_mod_list = par_unc_list + par_mod_list, []

    ## initialize MC dataset
    Par_mc = xr.Dataset()
    Par_mc.coords['config'] = np.arange(nMC)

    ## set random state
    rng = np.random.default_rng(seed)

    ## draw unc_ configurations
    for par in par_unc_list:
        assert len(Par[par].dims) == 1
        distrib = Par[par].dims[0].split('unc_')[-1]

        ## Normal distrib
        if distrib == 'Norm':
            mean, std = Par[par].sel(unc_Norm=['mean', 'std']).values
            mu, sigma = mean, abs(std)
            Norm = xr.DataArray(st.norm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = mu + sigma * Norm

        ## LogNormal distrib
        elif distrib == 'LogNorm':
            mean, std = Par[par].sel(unc_LogNorm=['mean', 'std']).values
            mu, sigma = lognorm_distrib_param(abs(mean), abs(std))
            Norm = xr.DataArray(st.norm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = np.sign(mean) * np.exp(mu + sigma * Norm)

        ## LogitNormal distrib
        elif distrib == 'LogitNorm':
            mean, std = Par[par].sel(unc_LogitNorm=['mean', 'std']).values
            mu, sigma = logitnorm_distrib_param(abs(mean), abs(std))
            if np.isnan([mu, sigma]).any(): raise RuntimeError('Could not infer LogitNorm distribution for parameter {}'.format(par)) 
            Norm = xr.DataArray(st.norm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = (1 + np.exp(mu + sigma * Norm)**-1)**-1

        ## two HalfNormal distribs
        elif distrib == '2HalfNorm':
            mean, std_neg, std_pos = Par[par].sel(unc_2HalfNorm=['mean', 'std_neg', 'std_pos']).values
            mu, sigma_neg, sigma_pos = mean, abs(std_neg), abs(std_pos)
            Bool = xr.DataArray(st.randint(0, 2).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            HalfNorm = xr.DataArray(st.halfnorm.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = mu + (Bool * sigma_pos - (1 - Bool) * sigma_neg) * HalfNorm

        ## Uniform distrib
        elif distrib == 'Uniform':
            mini, maxi = Par[par].sel(unc_Uniform=['mini', 'maxi']).values
            Uniform = xr.DataArray(st.uniform.rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = mini + (maxi - mini) * Uniform

        ## Triangle distrib
        elif distrib == 'Triangle':
            mode, mini, maxi = Par[par].sel(unc_Uniform=['mode', 'mini', 'maxi']).values
            Triang = xr.DataArray(st.triang(c=(mode-mini)/(maxi-mini)).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = mini + (maxi - mini) * Triang

        ## Discrete Uniform distrib
        elif distrib == 'Choice':
            mini, maxi = Par[par].sel(unc_Choice=['mini', 'maxi']).values
            mini, maxi = min(mini, maxi), max(mini, maxi)
            Choice = xr.DataArray(st.randint(mini, maxi + 1).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
            Par_mc[par] = Choice

        ## error otherwise
        else:
            raise RuntimeError("Distribution {} not implemented for parameter '{}'".format(distrib, par))

    ## draw mod_ configurations
    ## discrete draw of each mod
    Mod = xr.Dataset()
    for mod in mod_list:
        Mod[mod] = xr.DataArray(st.randint(0, len(Par[mod])).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config})
    ## applying selection (this keeps mod_ as secondary coordinate)
    Par_mod = xr.merge([Par[par] for par in par_mod_list])
    Par_mc = xr.merge([Par_mc, Par_mod.isel({mod: Mod[mod] for mod in mod_list})])
    ## adding noise based on Von Mises (if requested)
    if mod_noise > 0:
        for par in par_mod_list:
            Noise = xr.DataArray(st.vonmises_line(kappa=1/mod_noise**2).rvs(size=nMC, random_state=rng), coords={'config': Par_mc.config}) / np.pi
            Par_mc[par] *= 1 + Noise
        for mod in mod_list: del Par_mc[mod]

    ## draw kde_ configurations
    for par in par_kde_list: assert len(Par[par].dims) == 1
    for kde in kde_list:
        par_list = [par for par in par_kde_list if kde in Par[par].dims]
        kde_draw = st.gaussian_kde(np.array([Par[par].values for par in par_list]), bw_method=kde_bw).resample(nMC)
        for n, par in enumerate(par_list):
            Par_mc[par] = ('config', kde_draw[n, :])

    ## add parameters without uncertainty
    for par in [par for par in Par if par not in par_unc_list + par_mod_list + par_kde_list]:
        Par_mc[par] = Par[par]

    ## copy attributes
    for par in Par:
        Par_mc[par].attrs = Par[par].attrs

    ## return
    return Par_mc

