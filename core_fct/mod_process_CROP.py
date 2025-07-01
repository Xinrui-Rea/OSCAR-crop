##################################################
##################################################

"""
CONTENT
-------
CROP EMULATOR
1. YIELD
    1.1. CO2
    1.2. TEMPERATURE
    1.3. WATER
    1.4. FERTILIZER
2. LUC
    2.1. 
3. EMISSION
    3.1.
"""

##################################################
##################################################

import numpy as np
import xarray as xr

##################################################
##   CROP EMULATOR
##################################################
## initialize
def CROP(option='offline'):
    '''
    Options:
    ------
    option (str)        choose to run online or offline simulation

    '''
    
    if option == 'offline':
        from core_fct.cls_main import Model
        model = Model('OSCAR_v3_CROP')
    if option == 'online':
        from core_fct.mod_process import OSCAR
        model = OSCAR.copy(add_name='_CROP')
        
    ##################################################
    ## 1. YIELD
    ##################################################
    func0 = lambda x, a: a * x + 1
    func1 = lambda x, a: np.exp(a * x)
    func2 = lambda x, a, b: np.exp(-a*(x-b)**2+a*b**2)
    func3 = lambda x, a, b: 2/(np.exp(a*x)+np.exp(-b*x))
    func4 = lambda x, a, b: (np.exp(b) + 1) / (np.exp(-a*x+b) + 1)
    func5 = lambda x, a, b, c: (np.exp(-a*c)+np.exp(b*c))/(np.exp(a*(x-c))+np.exp(-b*(x-c)))
    func6 = lambda x, a, b, c: ((np.exp(b) + 1)/ (np.exp(-a*x+b) + 1))**c

    func_list = [func0, func1, func2, func3, func4, func5, func6]
    
    ## 2nd polynomial function without intercept
    func_2nd = lambda x, a, b: a*x**2+b*x
    ## Michaelis-Menten function
    func_MM = lambda x, a, b: a*x/(x+b)
    ## Mitcherlich function
    func_Mit = lambda x, a, b: a*np.expm1(b*x)
    ## George function
    func_Geo = lambda x, a, b: a*(np.power(0.99, x)-1)+b*x

    def number_func(number):
        return func_list[number]

    def func_number(func):
        return func_list.index(func)

    def find_params(func):
        param1 = [func0, func1]
        param2 = [func2, func3, func4]
        param3 = [func5, func6]
        if func in param1: num = 1
        if func in param2: num = 2
        if func in param3: num = 3
        return num

    ##===========
    ## 1.1. CO2
    ##===========
    ## CO2-yield response
    model.process('RC', ('D_CO2', ),
        lambda Var, Par: Eq__RC(Var, Par),
        units='1', core_dims=['spc_crop', 'reg_land'])

    def Eq__RC(Var, Par):
        RC = xr.full_like(Var.D_CO2, np.nan)
        for func_co2 in func_list:
            num_co2 = find_params(func_co2)
            params_co2 = [getattr(Par, f'{chr(97+i)}_CO2') for i in range(num_co2)]
            cond_co2 = Par.fct_CO2.astype(int) == func_list.index(func_co2)
            RC = xr.where(cond_co2, func_co2(Var.D_CO2/Par.CO2_0, *params_co2), RC)
        RC = RC.where(RC > 0)
        return RC

    ##===================
    ## 1.2. TEMPERATURE
    ##===================
    ## growing season temperature
    model.process('D_Tgs', ('D_Tl', ),
        lambda Var, Par: Eq__D_Tgs(Var, Par),
        units='K', core_dims=['spc_crop', 'reg_land'])

    def Eq__D_Tgs(Var, Par):
        return Par.w_Tgs*Var.D_Tl

    ## temperature-yield response
    model.process('RT', ('D_Tgs', ),
        lambda Var, Par: Eq__RT(Var, Par),
        units='1', core_dims=['spc_crop', 'reg_land'])

    def Eq__RT(Var, Par):
        RT = xr.ones_like(Var.D_Tgs)
        for func_temp in func_list:
            num_temp = find_params(func_temp)
            params_temp = [getattr(Par, f'{chr(97+i)}_Tgs') for i in range(num_temp)]
            cond_temp = Par.fct_Tgs.astype(int) == func_list.index(func_temp)
            RT = xr.where(cond_temp, func_temp(Var.D_Tgs, *params_temp), RT)
        RT = RT.where(RT > 0)
        return RT
    
    ##=============
    ## 1.3. PRECIPITATION
    ##=============
    ## growing season precipitation
    model.process('D_Pgs', ('D_Pl', ),
        lambda Var, Par: Eq__D_Pgs(Var, Par),
        units='mm yr-1', core_dims=['spc_crop', 'reg_land'])

    def Eq__D_Pgs(Var, Par):
        return Par.w_Pgs*Var.D_Pl

    ## precipitation-yield response of maize
    model.process('RP', ('D_Pgs', ),
        lambda Var, Par: Eq__RP(Var, Par),
        units='1', core_dims=['spc_crop', 'reg_land'])

    def Eq__RP(Var, Par):
        RP = xr.full_like(Var.D_Pgs, np.nan)
        for func_prec in func_list:
            num_prec = find_params(func_prec)
            params_prec = [getattr(Par, f'{chr(97+i)}_Pgs') for i in range(num_prec)]
            cond_prec = Par.fct_Pgs.astype(int) == func_list.index(func_prec)
            RP = xr.where(cond_prec, func_prec(Var.D_Pgs/Par.Pgs_0, *params_prec), RP)
        RP = xr.where(RP.irr == 'firr', 1, RP)
        RP = RP.where(RP > 0)
        return RP

    ##================
    ## 1.4. FERTILIZER
    ##================
    ## TODO: add biological fixation of soybean
    ## nitrogen input
    model.process('NI', ('N_fertl', 'N_dep'),
        lambda Var, Par: Eq__NI(Var, Par),
        units='kgN ha-1')
    
    def Eq__NI(Var, Par):
        return Var.N_fertl+Var.N_dep+Par.N_bnf
   
    ## nitrogen-yield response
    model.process('RN', ('NI', ),
        lambda Var, Par: Eq__RN(Var, Par),
        units='1')

    def Eq__RN(Var, Par):
        NI_0 = Par.N_fertl_0+Par.N_dep_0+Par.N_bnf
        RN = func_Geo(Var.NI / 100, Par.g_a, Par.g_b)/func_Geo(NI_0 / 100, Par.g_a, Par.g_b)
        RN = xr.where(RN.spc_crop == 'soy', 1, RN)
        RN = RN.where(RN > 0)
        return RN

    ##===========
    ## 1.5. YIELD
    ##===========
    ## maize yield
    model.process('YD', ('RC', 'RT', 'RP', 'RN'),
        lambda Var, Par: Eq__YD(Var, Par),
        units='tDM ha-1')
    
    def Eq__YD(Var, Par):
        return Par.YD_0*Var.RC*Var.RT*Var.RP*Var.RN
    
    ## RETURN
    return model