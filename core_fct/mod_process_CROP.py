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
    func1 = lambda x, a: a*x + 1
    func2 = lambda x, a: np.log1p(a*x) + 1
    func3 = lambda x, a, b: (1 + a*np.log1p(x)) / (1 + b*x)
    func4 = lambda x, a, b: -b*np.expm1(-a*x) + 1
    func5 = lambda x, a, b: np.exp(-a*(x-b)**2+a*b**2)
    func6 = lambda x, a, b: np.log1p(np.exp(-a*(x-b)**2 + a*b**2)) / np.log(2)
    func7 = lambda x, a, b: (np.exp(-a*b)+np.exp(a*b))/(np.exp(a*(x-b))+np.exp(-a*(x-b)))
    func8 = lambda x, a, b: (np.exp(b) + 1) / (np.exp(-a*x+b) + 1)
    func9 = lambda x, a, b, c: (np.exp(-a*b)+np.exp(c*b))/(np.exp(a*(x-b))+np.exp(-c*(x-b)))
    func10 = lambda x, a, b, c: ((np.exp(b) + 1)/ (np.exp(-a*x+b) + 1))**c
    
    ## 2nd polynomial function without intercept
    func_2nd = lambda a, b, x: a*x**2+b*x
    ## Michaelis-Menten function
    func_MM = lambda a, b, x: a*x/(x+b)
    ## Mitcherlich function
    func_Mit = lambda a, b, x: a*np.expm1(b*x)
    ## George function
    func_Geo = lambda a, b, x: a*(np.power(0.99, x)-1)+b*x
    
    ##===========
    ## 1.1. CO2
    ##===========
    ## CO2-yield response
    model.process('RC', ('D_CO2', ),
        lambda Var, Par: Eq__RC(Var, Par),
        units='1', core_dims=['spc_crop', 'reg_land'])

    def Eq__RC(Var, Par):
        RC1 = func1(Var.D_CO2/Par.CO2_0, Par.a_CO2)
        RC2 = func2(Var.D_CO2/Par.CO2_0, Par.a_CO2)
        RC3 = func3(Var.D_CO2/Par.CO2_0, Par.a_CO2, Par.b_CO2)
        RC4 = func4(Var.D_CO2/Par.CO2_0, Par.a_CO2, Par.b_CO2)
        RC5 = func5(Var.D_CO2/Par.CO2_0, Par.a_CO2, Par.b_CO2)
        RC6 = func6(Var.D_CO2/Par.CO2_0, Par.a_CO2, Par.b_CO2)
        RC7 = func7(Var.D_CO2/Par.CO2_0, Par.a_CO2, Par.b_CO2)
        RC8 = func8(Var.D_CO2/Par.CO2_0, Par.a_CO2, Par.b_CO2)
        RC = xr.concat([RC1.assign_coords(fct_YD=1), RC2.assign_coords(fct_YD=2), RC3.assign_coords(fct_YD=3),
            RC4.assign_coords(fct_YD=4), RC5.assign_coords(fct_YD=5), RC6.assign_coords(fct_YD=6), 
            RC7.assign_coords(fct_YD=7), RC8.assign_coords(fct_YD=8)], dim='fct_YD')
        return (Par.RC_switch*RC).sum('fct_YD', min_count=1)

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
        RT5 = func5(Var.D_Tgs, Par.a_Tgs, Par.b_Tgs)
        RT6 = func6(Var.D_Tgs, Par.a_Tgs, Par.b_Tgs)
        RT7 = func7(Var.D_Tgs, Par.a_Tgs, Par.b_Tgs)
        RT8 = func8(Var.D_Tgs, Par.a_Tgs, Par.b_Tgs)
        RT9 = func9(Var.D_Tgs, Par.a_Tgs, Par.b_Tgs, Par.c_Tgs)
        RT10 = func10(Var.D_Tgs, Par.a_Tgs, Par.b_Tgs, Par.c_Tgs)
        RT = xr.concat([RT5.assign_coords(fct_YD=5), RT6.assign_coords(fct_YD=6), RT7.assign_coords(fct_YD=7), 
            RT8.assign_coords(fct_YD=8), RT9.assign_coords(fct_YD=9), RT10.assign_coords(fct_YD=10)], dim='fct_YD')
        return (Par.RT_switch * RT).sum('fct_YD', min_count=1)

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
        RP5 = func5(Var.D_Pgs/Par.Pgs_0, Par.a_Pgs, Par.b_Pgs).where(Par.a_Pgs.irr == 'noirr', 1)
        RP6 = func6(Var.D_Pgs/Par.Pgs_0, Par.a_Pgs, Par.b_Pgs).where(Par.a_Pgs.irr == 'noirr', 1)
        RP7 = func7(Var.D_Pgs/Par.Pgs_0, Par.a_Pgs, Par.b_Pgs).where(Par.a_Pgs.irr == 'noirr', 1)
        RP8 = func8(Var.D_Pgs/Par.Pgs_0, Par.a_Pgs, Par.b_Pgs).where(Par.a_Pgs.irr == 'noirr', 1)
        RP9 = func9(Var.D_Pgs/Par.Pgs_0, Par.a_Pgs, Par.b_Pgs, Par.c_Pgs).where(Par.a_Pgs.irr == 'noirr', 1)
        RP10 = func10(Var.D_Pgs/Par.Pgs_0, Par.a_Pgs, Par.b_Pgs, Par.c_Pgs).where(Par.a_Pgs.irr == 'noirr', 1)
        RP = xr.concat([RP5.assign_coords(fct_YD=5), RP6.assign_coords(fct_YD=6), RP7.assign_coords(fct_YD=7), 
            RP8.assign_coords(fct_YD=8), RP9.assign_coords(fct_YD=9), RP10.assign_coords(fct_YD=10)], dim='fct_YD')
        return (Par.RP_switch * RP).sum('fct_YD', min_count=1)

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
        RN_2nd = func_2nd(Par.g_a, Par.g_b, Var.NI)/func_2nd(Par.g_a, Par.g_b, NI_0)
        RN_MM = func_MM(Par.g2_a, Par.g2_b, Var.NI)/func_MM(Par.g2_a, Par.g2_b, NI_0)
        RN_Mit = func_Mit(Par.g3_a, Par.g3_b, Var.NI)/func_Mit(Par.g3_a, Par.g3_b, NI_0)
        RN_Geo = func_Geo(Par.g4_a, Par.g4_b, Var.NI)/func_Geo(Par.g4_a, Par.g4_b, NI_0)
        return Par.RN_is_2nd*RN_2nd+Par.RN_is_MM*RN_MM+Par.RN_is_Mit*RN_Mit+Par.RN_is_Geo*RN_Geo
    
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