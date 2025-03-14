import numpy as np

METEO_FRANCE_LIST = ["TAL", "CHE", "NOI", "YEU", "NAZ"]
DICT_EXPE_DATE = {
#   code  : index, nom,                        lat °N,  lon °E,  grandeur, [start-end] date
    "CAR" : ( 1, "CARDINAUX WM",              [47.321,  -2.835], ("U"), ["2016-04-01", "2018-06-01", "2021-03-01", "2022-04-01"], [0.4, 0.4, 1] ),
    "TEI" : ( 2, "TEIGNOUSE WM",              [47.457,  -3.046], ("U"), ["2016-09-01","2018-10-30", "2019-01-15","2020-02-28", "2020-11-01","2021-02-28", "2021-04-01", "2022-04-01"], [1, 0, 0]),
    "TAL" : ( 3, "TALUT ECN",                 [47.294,  -3.218], ("U"), ["2023-03-08", "2023-07-30"], [0.7, 0.9, 0.5] ),
    "CHE" : ( 4, "CHEMOULIN MF",              [47.233,  -2.298], ("U"), ["2020-01-01", "2023-12-31"], [0.8, 0.6, 0.4] ),
    "NOI" : ( 5, "NOIRMOUTIER MF",            [47.004,  -2.257], ("U"), ["2020-01-01", "2023-12-31"], [0.8, 0.6, 0.4] ),
    "YEU" : ( 6, "YEU MF",                    [46.693,  -2.330], ("U"), ["2020-01-01", "2023-12-31"], [0.8, 0.6, 0.4] ),
    "PEN" : ( 7, "PEN MEN WM",                [47.648,  -3.510], ("U"), None, None ),
    "KER" : ( 8, "KERROCH WM",                [47.700,  -3.461], ("U"), None, None ),
    "CRO" : ( 9, "CROISIC ECN",               [47.273,  -2.517], ("U"), ["2020-03-01", "2020-09-30"], [0.7, 0.9, 0.5] ),
    "BI1" : (10, "LIDAR BELLE-ILE NORD DGEC", [47.405,  -3.570], ("U"), ["2020-07-24", "2021-10-09"], [1, 0.8, 0.2] ),
    "BI2" : (11, "LIDAR BELLE-ILE SUD DGEC",  [47.217,  -3.500], ("U"), ["2020-07-24", "2021-10-08"], [1, 0.8, 0.2] ),
    "BI3" : (12, "BOUEE BELLE-ILE",           [0.0000,  0.0000], ("U"), None, None ),
    "IST" : (13, "ISTHME QUIBERON WM",        [47.551,  -3.135], ("U"), None, None ),
    "ENV" : (14, "ENVSN WM",                  [47.510,  -3.119], ("U"), None, None ),
    "NAV" : (15, "PORT NAVALO WM",            [47.548,  -2.918], ("U"), None, None ),
    "ARZ" : (16, "ILE D'ARZ WM",              [47.595,  -2.810], ("U"), None, None ),
    "DUM" : (17, "ILE DUMET WM",              [47.412,  -2.620], ("U"), None, None ),
    "SEM" : (18, "SEMREV",                    [47.238,  -2.786], ("T"), None, None ),
    "ETL" : (19, "SEMAPHORE D'ETEL WM",       [47.646,  -3.214], ("U"), None, None ),
    "TA2" : (20, "TALUT MF",                  [47.294,  -3.218], ("U"), ["2020-01-01", "2023-12-31"], [0.8, 0.6, 0.4] ),
    "NAZ" : (21, "ST NAZAIRE AERO",           [47.314,  -2.154], ("U"), None, None ),
    "DIN" : (32, "DINARD AERO",               [48.585,  -2.076], ("U"), None, None ),
    "LI1" : (34, "LIDAR CROISIC",             [47.273,  -2.516], ("RWS"), None, None ),
    "ARO" : (80, "DONNEES AROME TELEM",       [47.294,  -3.218], ("U"), ["2016-01-01", "2022-12-31"], [0.6, 0.8, 1] ),
    "CP1" : (96, "CRO TERRE +20KM",           [47.400,  -2.328], ("U"), None, None ),
    "CP2" : (97, "CRO TERRE +40KM",           [47.527,  -2.140], ("U"), None, None ),
    "CM1" : (98, "CRO MER -20KM",             [47.145,  -2.703], ("U"), None, None ),
    "CM2" : (99, "CRO MER -40KM",             [47.018,  -2.889], ("U"), None, None ),
}

#https://www.loire-atlantique.gouv.fr/contenu/telechargement/49345/321410/file/2017%2002%20Annexe%20-%20Dossier%20de%20pr%C3%A9cisions%20techniques%20convention%20DPM%20PBG.pdf
PARC_ST_NAZAIRE = np.array([
    [47.21, -2.66], #A
    [47.203, -2.692], #B
    [47.183, -2.687], #C
    [47.147, -2.7], #D
    [47.125, -2.58], #E
    [47.113, -2.565], #F
    [47.148, -2.498], #G
    [47.172, -2.52], #H
    [47.183, -2.57], #I
    [47.163, -2.58], #J
    [47.178, -2.635], #K
    [47.21, -2.66] #A (to close the loop)
])

SONIC_VARNAMES = {
    "TSON"      : "T", #Sonic Air Temperature
    "U"         : "U",
    "V"         : "V",
    "W"         : "W",
    "T"         : "Temp", #Sensor Air Temperature
    "WD"        : "WindDir_H",
    "MH"        : "WindSpeed_H",
    "AVAIL"     : "Availability",
    "T_MIN"     : "T_Min",
    "T_MAX"     : "T_Max",
    "U_MIN"     : "U_Min",
    "U_MAX"     : "U_Max",
    "V_MIN"     : "V_Min",
    "V_MAX"     : "V_Max",
    "W_MIN"     : "W_Min",
    "W_MAX"     : "W_Max",
    "T_MIN"     : "Temp_Min",
    "T_MAX"     : "Temp_Max",
    "MH_MIN"    : "WindSpeed_H_Min",
    "WD_MIN"    : "WindSpeed_H_Max",
    "M2TSON"    : "COV_T_T",
    "COVTTSON"  : "COV_T_Temp",
    "COVUTSON"  : "COV_T_U",
    "COVVTSON"  : "COV_T_V",
    "COVWTSON"  : "COV_T_W",
    "COVMHTSON" : "COV_T_WindSpeed_H",
    "M2T"       : "COV_Temp_Temp",
    "COVUT"     : "COV_Temp_U",
    "COVVT"     : "COV_Temp_V",
    "COVWT"     : "COV_Temp_W",
    "COVMHT"    : "COV_Temp_WindSpeed_H",
    "M2U"       : "COV_U_U",
    "COVUV"     : "COV_U_V",
    "COVUW"     : "COV_U_W",
    "COVUMH"    : "COV_U_WindSpeed_H",
    "M2V"       : "COV_V_V",
    "COVVW"     : "COV_V_W",
    "COVVMH"    : "COV_V_WindSpeed_H",
    "M2W"       : "COV_W_W",
    "COVWMH"    : "COV_W_WindSpeed_H",
    "M2H"       : "COV_WindSpeed_H_WindSpeed_H",
    "ORIGIN"    : "origin",   
}
