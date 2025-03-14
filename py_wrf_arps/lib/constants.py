#WRF/share/module_modelants.F
EARTH_RADIUS = 6370000.0 # in meters (reradius)
OMEGA = 7.2921e-5 # rad/s (EOMEG)
G = 9.81 #m.s-2 (g)
RD = 287.0 #J/(kg.K)
CP = 7.0*RD/2.0 #
RDDCP = 2.0/7.0 # Rd/Cp
P0 = 1e5 #Pa 
CELSIUS = 273.15 #
RV = 461.6 #J/(kg.K)
LV = 2.5e6 #J/kg (Stull 2017 p.88)
E0 = 611.3 #Pa (Stull 2017 p.88)
GAMMA = 1.4
T0 = 300 #K
BETA = G/T0 #m.s-2.K-1 (= 0.033)
KARMAN = 0.4 
CS = 0.25 # Smagorinsky constant
CK = 0.15 # TKE constant
EPSILON = 0.61
EPSILON2 = 0.622 #(Stull 2017 p.88)

#Copied from WRF source code (phys/module_bl_myjpbl, lines 41-45)
#See Janjic (2002), Mellor and Yamada (1982)
A1_MYJ = 0.659888514560862645
A2_MYJ = 0.6574209922667784586
B1_MYJ = 11.87799326209552761
B2_MYJ = 7.226971804046074028
C1_MYJ = 0.000830955950095854396
BETA_MYJ = 1/273

def Kelvin_to_Celsius(K):
    return K - CELSIUS

def Celsius_to_Kelvin(C):
    return C + CELSIUS

def T_to_PT(T, P):
    return T*(P0/P)**RDDCP

def PT_to_T(PT, P):
    return PT*(P0/P)**-RDDCP

def RHO_to_PRHO(RHO, P): 
    return RHO*(P0/P)**(1-RDDCP)

def PRHO_to_RHO(PRHO, P):
    return PRHO*(P0/P)**(RDDCP-1)