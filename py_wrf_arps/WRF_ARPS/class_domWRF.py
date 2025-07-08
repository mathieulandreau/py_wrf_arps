#!/usr/bin/env python3
import sys
from .class_dom import Dom
from ..class_variables import VariableSave, Variable
from ..lib import constants, manage_time, manage_projection, manage_dict

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from wrf import vertcross, CoordPair, interplevel, WrfProj
import collections #to sort dictionnaries
import copy #for dictionnaries kwargs in calculate
import datetime

debug=False

#BE CAREFUL, IN WRFinput and WRF, THE BASE STATE TEMPERATURE IS T0 = 300K and NOT T00 = (chosen value in namelist.input / base temperature)
class DomWRF(Dom):
    software = "WRF"
    suffix_length = 19 #2020-05-14_00:00:00
    # Time avg values that can be calculated thanks to accumulated variables
    avg_from_ac_list = ["SH_FLX_AVG", "LH_FLX_AVG", "SWUPT_AVG", "SWUPTC_AVG", "SWDNT_AVG",\
                        "SWDNTC_AVG", "SWUPB_AVG", "SWUPBC_AVG", "SWDNB_AVG", "SWDNBC_AVG",\
                        "LWUPT_AVG", "LWUPTC_AVG", "LWDNT_AVG", "LWDNTC_AVG", "LWUPB_AVG",\
                        "LWUPBC_AVG", "LWDNB_AVG", "LWDNBC_AVG", "HFX_AVG"]
    #file from time series
    dict_varname_ts = {
        "U" : "UU",
        "V" : "VV",
        "W" : "WW",
        "ZP" : "PH",
        "P" : "PR",
        "QV" : "QV",
        "PT" : "TH",
        "T2" : ["TS", 5],
        "QV2" : ["TS", 6],
        "U10" : ["TS", 7],
        "V10" : ["TS", 8],
        "P_SFC" : ["TS", 9],
        "LW" : ["TS", 10], #ground long waves (downward = positive)
        "SW" : ["TS", 11], #ground short waves
        "SH_FLX" : ["TS", 12], #sensible heat flux
        "LH_FLX" : ["TS", 13], #latent heat flux
        "T_SKIN" : ["TS", 14], 
        "T_SOIL" : ["TS", 15], #1st layer soil temp
        "RAINC" : ["TS", 16], #rainfall from cumulus scheme
        "RAINNC" : ["TS", 17], #rainfall from explicit scheme
        "CLW" : ["TS", 18], #total column integrated water vapor and cloud var
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_output_filenames(self):
        self.output_filenames = {
            "hist" : {},
            "base" : {},
            "post" : {},
            "post_static" : {},
            "df" : {},
            "diag" : {},
            "diag2" : {},
            "stats" : {},
            "traj" : {},
            "ts" : {},
        }             
        for filename in os.listdir(self.output_data_dir):
            if filename.startswith(self.output_prefix) :
                if self.keep_open :
                    self.output_filenames['hist'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r") #main output file
                else :
                    self.output_filenames['hist'][self.output_data_dir + filename] = True
                self.FLAGS['hist'] = True
            elif filename.startswith("wrf_static_d"+self.i_str) :
                if self.keep_open :
                    self.output_filenames['base'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r") #main static variables
                else :
                    self.output_filenames['base'][self.output_data_dir + filename] = True
                self.FLAGS['base'] = True
            elif filename.startswith("wrflux_d"+self.i_str) :
                if self.keep_open :
                    self.output_filenames['stats'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r") #wrflux variables
                else :
                    self.output_filenames['stats'][self.output_data_dir + filename] = True
                self.FLAGS['stats'] = True
            elif filename.startswith("output_diagnostics_d"+self.i_str) :
                if self.keep_open :
                    self.output_filenames['diag'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r") #if output_diagnostics (must rename auxhist_...)
                else :
                    self.output_filenames['diag'][self.output_data_dir + filename] = True
                self.FLAGS['diag'] = True
            elif filename.startswith("diag_nwp2_d"+self.i_str) :
                if self.keep_open :
                    self.output_filenames['diag2'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r") #if diag_nwp2 (must rename auxhist_...)
                else :
                    self.output_filenames['diag2'][self.output_data_dir + filename] = True
                self.FLAGS['diag2'] = True
            elif filename.startswith("wrfout_traj_d"+self.i_str) :
                if self.keep_open :
                    self.output_filenames['traj'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r") #trajectory files
                else :
                    self.output_filenames['traj'][self.output_data_dir + filename] = True
                self.FLAGS['traj'] = True
            elif filename[-2:].isalpha() and filename[-3]=="." and filename[-5:-3].isnumeric() and filename[-7:-5] == ".d" : #time series files
                pfx = filename[:-7]
                if pfx not in self.output_filenames['ts'] :
                    self.output_filenames['ts'][pfx] = []
                self.output_filenames['ts'][pfx].append(self.output_data_dir + filename)
                self.FLAGS['ts'] = True
        for filename in os.listdir(self.postprocdir):
            if filename.startswith(self.name + "_post_static" ):
                self.FLAGS['post_static'] = True
                # if False:
                if self.keep_open :
                    self.output_filenames['post_static'][self.postprocdir + filename] = Dataset(self.postprocdir + filename, "r")
                else :
                    self.output_filenames['post_static'][self.postprocdir + filename] = True
            elif filename.startswith(self.name + "_post_" ):
                self.FLAGS['post'] = True
                # if False:
                if self.keep_open :
                    self.output_filenames['post'][self.postprocdir + filename] = Dataset(self.postprocdir + filename, "r")
                else :
                    self.output_filenames['post'][self.postprocdir + filename] = True
            if filename.startswith(self.name + "_df_TIME.pkl" ):
                self.output_filenames['df'][self.postprocdir + filename] = pd.read_pickle(self.postprocdir + filename)
                self.FLAGS['df'] = True
        
        if not self.FLAGS['hist'] :
            raise(Exception("error in "+self.name+", no file starting with "+self.output_prefix+" found in "+self.output_data_dir))
        # sort 
        self.output_filenames['hist'] = collections.OrderedDict(sorted(self.output_filenames['hist'].items()))
        self.output_filenames['stats'] = collections.OrderedDict(sorted(self.output_filenames['stats'].items()))
        self.output_filenames['diag'] = collections.OrderedDict(sorted(self.output_filenames['diag'].items()))
        self.output_filenames['diag2'] = collections.OrderedDict(sorted(self.output_filenames['diag2'].items()))
        self.output_filenames['post'] = collections.OrderedDict(sorted(self.output_filenames['post'].items()))
        
    def init_saved_variables(self):
        if self.FLAGS['base'] :
            key = 'base' #for dom_WRFinput and new wrf settings when static data are in user-defined wrf_static
        else :
            key = 'hist' #for old WRF settings when everything was in wrfout
        
        filename = next(iter(self.output_filenames[key]))
        print("initializing from", filename)
        if self.keep_open :
            file = self.output_filenames[key][filename]
        else :
            file = Dataset(filename, "r")
        
        #Getting terrain height nad landmask
        self.VARIABLES['HT'].value = HT = file['HGT'][:][0]
        self.VARIABLES['LANDMASK'].value = file['LANDMASK'][:][0]

        #Getting simulation initial time
        self.VARIABLES['INITIAL_TIME'].value = self.str2date(file.START_DATE, self.software)
        self.VARIABLES["SIM_INITIAL_TIME"] = VariableSave("", "", "", "", 0, self.str2date(self.get_data("SIMULATION_START_DATE"), self.software))
        #Getting projection and central coordinates
        self.VARIABLES['MAPPROJ'].value = MAPPROJ = file.MAP_PROJ
        self.VARIABLES['CTRLAT'].value = CTRLAT = file.CEN_LAT
        self.VARIABLES['CTRLON'].value = CTRLON = file.CEN_LON

        #Getting dimensions
        self.VARIABLES['NX_XSTAG'].value = NX_XSTAG = file.dimensions["west_east_stag"].size
        self.VARIABLES['NY_YSTAG'].value = NY_YSTAG = file.dimensions["south_north_stag"].size
        self.VARIABLES['NZ_ZSTAG'].value = NZ_ZSTAG = file.dimensions["bottom_top_stag"].size
        self.VARIABLES['NZ_SOIL_ZSTAG'].value = NZ_SOIL_ZSTAG = file.dimensions["soil_layers_stag"].size + 1
        self.VARIABLES['NX'].value = NX = NX_XSTAG - 1
        self.VARIABLES['NY'].value = NY = NY_YSTAG - 1
        self.VARIABLES['NZ'].value = NZ = NZ_ZSTAG - 1
        self.VARIABLES['NZ_SOIL'].value = NZ_SOIL = NZ_SOIL_ZSTAG - 1

        #Getting Latitude and Longitude
        self.VARIABLES['LAT'].value = LAT = file['XLAT'][:][0]
        self.VARIABLES['LON'].value = LON = file['XLONG'][:][0]  
        self.VARIABLES['LAT_XSTAG'].value = LAT_XSTAG = file['XLAT_U'][:][0]
        self.VARIABLES['LON_XSTAG'].value = LON_XSTAG = file['XLONG_U'][:][0]
        self.VARIABLES['LAT_YSTAG'].value = LAT_YSTAG = file['XLAT_V'][:][0]
        self.VARIABLES['LON_YSTAG'].value = LON_YSTAG = file['XLONG_V'][:][0]
        self.VARIABLES['ETA'].value = ETA = file['ZNU'][:][0]
        self.VARIABLES['ETA_ZSTAG'].value = ETA_ZSTAG = file['ZNW'][:][0]
        ETA3 = np.tensordot(ETA, np.ones((NY, NX)), 0)
        self.VARIABLES["ETA3"] = VariableSave("", "", "", "", 0, ETA3) 
        ETA3_ZSTAG = np.tensordot(ETA_ZSTAG, np.ones((NY, NX)), 0)
        self.VARIABLES["ETA3_ZSTAG"] = VariableSave("", "", "", "", 0, ETA3_ZSTAG) 


        #Setting the chosen projection to match between all simulations
        self.VARIABLES['TRUELAT1'].value = manage_projection.TRUELAT1
        self.VARIABLES['TRUELAT2'].value = manage_projection.TRUELAT2
        self.VARIABLES['TRUELON'].value = manage_projection.TRUELON
        self.VARIABLES['FALSE_EASTING'].value = manage_projection.FALSE_EASTING 
        self.VARIABLES['FALSE_NORTHING'].value = manage_projection.FALSE_NORTHING
        CRS = manage_projection.CRS
        self.VARIABLES['CRS'] = VariableSave("", "", "", "", 0, CRS)

        #Calculating new X, Y with chosen projection
        self.VARIABLES['X'].value, self.VARIABLES['Y'].value = manage_projection.ll_to_xy(LON, LAT, CRS)
        self.VARIABLES['X_XSTAG'].value, self.VARIABLES['Y_XSTAG'].value = manage_projection.ll_to_xy(LON_XSTAG, LAT_XSTAG, CRS)
        self.VARIABLES['X_YSTAG'].value, self.VARIABLES['Y_YSTAG'].value = manage_projection.ll_to_xy(LON_YSTAG, LAT_YSTAG, CRS)
        self.VARIABLES['DX'].value = file.DX
        self.VARIABLES['DY'].value = file.DY
        self.VARIABLES['DX_PROJ'].value = file.DX
        self.VARIABLES['DY_PROJ'].value = file.DY
        self.VARIABLES['DT'].value = datetime.timedelta(seconds = float(file.DT))

        #Getting Z mesh
        Z_SOIL_un = file['ZS'][:][0]
        Z_SOIL = np.zeros((NZ_SOIL, NY, NX))
        ZP_SOIL = np.zeros((NZ_SOIL, NY, NX))
        Z_SOIL_ZSTAG = np.zeros((NZ_SOIL_ZSTAG, NY, NX))
        ZP_SOIL_ZSTAG = np.zeros((NZ_SOIL_ZSTAG, NY, NX))
        for k in range(1, NZ_SOIL):
            Z_SOIL[k, :, :] = Z_SOIL_un[k-1]
            ZP_SOIL[k, :, :] = Z_SOIL[k, :, :] + HT
            Z_SOIL_ZSTAG[k+1, :, :] = 2 * Z_SOIL[k, :, :] - Z_SOIL_ZSTAG[k, :, :] #probably useless
            ZP_SOIL_ZSTAG[k+1, :, :] = 2 * ZP_SOIL[k, :, :] - ZP_SOIL_ZSTAG[k, :, :] #probably useless

        self.VARIABLES['Z_SOIL'].value = Z_SOIL
        self.VARIABLES['ZP_SOIL'].value = ZP_SOIL
        self.VARIABLES['Z_SOIL_ZSTAG'].value = Z_SOIL_ZSTAG
        self.VARIABLES['ZP_SOIL_ZSTAG'].value = ZP_SOIL_ZSTAG
        self.VARIABLES['DZ_SOIL'].value = file['DZS'][:][0]

        #Setting bas state temperature
        T0 = 300.0 #BE CAREFUL, IN WRFinput and WRF, THE BASE STATE TEMPERATURE IS T0 = 300K and NOT T00 = (chosen value in namelist.input / base temperature)
        self.VARIABLES["T0"] = VariableSave("", "", "", "", 0, T0) 
        self.VARIABLES["PT_BASE"].value = T0 * np.ones((NZ, NY, NX))
        self.VARIABLES["PTV_BASE"].value = T0 * np.ones((NZ, NY, NX))
        self.VARIABLES["QV_BASE"].value = np.zeros((NZ, NY, NX))
        #HT3 is the terrain Height (2D vector), copied and paste all along the Z dimension (resulting in a 3D array)
        #It is used to swith from Z to ZP
        HT3_ZSTAG = np.tensordot(np.ones(NZ_ZSTAG), HT, 0)
        HT3 = np.tensordot(np.ones(NZ), HT, 0)
        self.VARIABLES["HT3"] = VariableSave("", "", "", "", 0, HT3) 
        self.VARIABLES["HT3_ZSTAG"] = VariableSave("", "", "", "", 0, HT3_ZSTAG) 
        self.VARIABLES['NMK'].value = NMK = 0
        
        if not self.keep_open :
            file.close()
    
    def init_hardcoded_variables(self) :
        for varname in self.avg_from_ac_list :
            if not varname in self.VARIABLES :
                self.VARIABLES[varname] = Variable(varname, varname, "$W.m^{-2}$", "W.m-2", 2, cmap=self.get_cmap("AC"+varname[:-4]))
            
    def calculate(self, varname, **kwargs):
        if varname in ["U_PERT", "V_PERT", "W_PERT", "M_PERT", "MH_PERT", "WD_PERT", "QV_PERT"]: #BASE = 0 => PERT = Total
            return self.get_data(varname[:-5], **kwargs)
        elif varname in ["U_BASE", "V_BASE", "W_BASE", "M_BASE", "MH_BASE", "WD_BASE", "QV_BASE"]: #BASE = 0
            return self.get_data("ZERO", **kwargs)
        elif varname in ["U_PERT_XSTAG", "V_PERT_YSTAG","W_PERT_ZSTAG"]: #BASE = 0 => PERT = Total
            return self.get_data(varname[:-11] + varname[-6:], **kwargs)
        elif varname in ["U_BASE_XSTAG", "V_BASE_YSTAG","W_BASE_ZSTAG"]: #BASE = 0
            return self.get_data("ZERO"+varname[-6:], **kwargs)
        elif varname in ["U"]:
            if kwargs["avg"] :
                return self.get_data("U_AVG", **kwargs)
            else : #unstag X
                kwargs_temp = copy.copy(kwargs)
                kwargs_temp["i_unstag"] = 2
                return self.get_data(varname+"_XSTAG", **kwargs_temp)
        elif varname in ["V"]:
            if kwargs["avg"] :
                return self.get_data("V_AVG", **kwargs)
            else : #unstag Y
                kwargs_temp = copy.copy(kwargs)
                kwargs_temp["i_unstag"] = 1
                return self.get_data(varname+"_YSTAG", **kwargs_temp)
        elif varname in ["W_INST"]:
            new_kwargs = copy.copy(kwargs)
            new_kwargs["saved"] = {}
            new_kwargs["avg"] = False
            return self.get_data("W", **new_kwargs)
        elif varname in ["W", "PHI", "PHI_BASE", "PHI_PERT"]:
            if kwargs["avg"] and varname == "W" :
                return self.get_data("W_AVG", **kwargs)
            elif kwargs["avg"] and varname == "PHI" :
                return constants.G * self.get_data("ZP", **kwargs)
            elif kwargs["avg"] and varname == "PHI_PERT" :
                return constants.G * self.get_data("ZP_PERT", **kwargs)
            else : #unstag Z
                kwargs_temp = copy.copy(kwargs)
                kwargs_temp["i_unstag"] = 0
                return self.get_data(varname+"_ZSTAG", **kwargs_temp)
        elif varname in ["P", "PTV", "MUD", "MUT", "PID"]:  #BASE + PERT
            var_BASE = self.get_data(varname+"_BASE", **kwargs)
            var_PERT = self.get_data(varname+"_PERT", **kwargs)
            return var_BASE + var_PERT
        elif varname in ["PI"]:  #BASE + PERT
            var_BASE = self.get_data("P_BASE", **kwargs)
            var_PERT = self.get_data("PI_PERT", **kwargs)
            return var_BASE + var_PERT
        elif varname in ["PT"] :
            PTV = self.get_data("PTV", **kwargs)
            QV = self.get_data("QV", **kwargs)
            return PTV/(1+constants.RV/constants.RD * QV)
        elif varname in ["T_BASE"] : 
            return self.get_data("PT_BASE", **kwargs) #T0
        elif varname in ["T", "TV"]: #convert potential temperature to real temperature
            PT_var = self.get_data("P"+varname, **kwargs)
            P = self.get_data("P", **kwargs)
            return constants.PT_to_T(PT_var, P)
        elif varname in ["T2"]: #convert potential temperature to real temperature
            PT_var = self.get_data("P"+varname, **kwargs)
            if kwargs["crop"][0] in [0, [0, 1]] :
                new_kwargs = kwargs
            else :
                new_kwargs = copy.deepcopy(kwargs)
                new_kwargs["saved"] = {}
                new_kwargs["crop"] = ([0,1], kwargs["crop"][1], kwargs["crop"][2])
            zaxis = self.find_axis("z", dim=3, varname="P", **new_kwargs)
            print(zaxis)
            P = self.get_data("P", **new_kwargs)
            print(P.shape)
            P = np.squeeze(P, axis=zaxis)
            print(PT_var.shape, P.shape)
            return constants.PT_to_T(PT_var, P)
        elif varname in ["ZP_PERT", "MUT_PERT", "PTV_PERT", "P_PERT", "PI_PERT"]:
            if kwargs["avg"] :
                return self.get_data(varname+"_AVG", **kwargs)
            else :
                raise(Exception(varname+" should be Read variables. If you are here, it means that "+varname+" was not found so you should activate avg = True"))
        elif varname in ["T_PERT", "PT_PERT"]: 
            if kwargs["avg"] :
                return self.get_data(varname)
            else :
                var_TOTAL = self.get_data(varname[:-5], **kwargs)
                var_BASE = self.get_data(varname[:-5]+"_BASE", **kwargs)
                return var_TOTAL - var_BASE
        elif varname == "T_SOIL":
            #WARNING, if z dimension is cropped, NZ_SOIL will be wrong
            # DO NOT USE WITH Z CROP, ALWAYS USE WITH NO CROP OR WITH CROP=("ALL", .., ..)
            LANDMASK = self.get_data("LANDMASK", **kwargs)
            T_SOIL = self.get_data("TSLB", **kwargs)
            T_SKIN = self.get_data("T_SKIN", **kwargs)
            pos = np.where(LANDMASK == 0)
            NZ_SOIL = self.get_data("NZ_SOIL")
            if "crop" in kwargs :
                if type(kwargs["crop"][0]) is int :
                    T_SOIL[pos] = T_SKIN[pos]
                    return T_SOIL
                elif type(kwargs["crop"][0]) is list :
                    NZ_SOIL = kwargs["crop"][0][1] - kwargs["crop"][0][0]
            for k in range(NZ_SOIL): #Same as WRF2ARPS
                T_SOIL[k][pos] = T_SKIN[pos]
            return T_SOIL
        elif varname in ["RH"] : #WARNING : doesn't take into account snow, rain or ice
            #Warning : QV here is mixing ratio whereas in Stull 2017 p.88, q is specific humidity and r is mixing ratio
            T = self.get_data("T", **kwargs)
            QV = self.get_data("QV", **kwargs)
            P = self.get_data("P", **kwargs)
            es = constants.E0 * np.exp(constants.LV/constants.RV * (1/273.15 - 1/T))
            e = QV*P/(constants.EPSILON2-QV) #Mixing ratio in Stulle 2017 p.88
            return e/es
        elif varname in ["RH100"] :
            return self.get_data("RH", **kwargs)*100
        elif varname in ["PHI_ZSTAG", "PID_ZSTAG", "MUD_ZSTAG"]: #BASE + PERT Z_STAG
            if kwargs["avg"] and varname == "PHI_ZSTAG" :
                return constants.G * self.get_data("ZP_ZSTAG", **kwargs)
            else :
                var_unstag = varname[:-6]
                var_BASE_ZSTAG = self.get_data(var_unstag + "_BASE_ZSTAG", **kwargs)
                var_PERT_ZSTAG = self.get_data(var_unstag + "_PERT_ZSTAG", **kwargs) 
                return var_BASE_ZSTAG + var_PERT_ZSTAG
        elif varname in ["ZP"] :
            if kwargs["avg"] :
                ZP_BASE = self.get_data("ZP_BASE", **kwargs)
                ZP_PERT = self.get_data("ZP_PERT", **kwargs)
                return ZP_BASE + ZP_PERT
            else :
                var_PHI = self.get_data("PHI", **kwargs)
                return var_PHI/constants.G
        elif varname in ["ZP_ZSTAG"] :
            if kwargs["avg"] :
                NZ_ZSTAG = self.get_data("NZ_ZSTAG")
                NX = self.get_data("NX")
                NY = self.get_data("NY")
                HT = self.get_data("HT")
                if "crop" in kwargs :
                    crop = kwargs["crop"]
                else : 
                    crop = ("ALL", "ALL", "ALL")
                tab_slice, _, _, _ = self.prepare_crop_unstag(crop, 1, (), None, (NZ_ZSTAG, NY, NX))
                new_crop = ("ALL", "ALL", "ALL")
                new_kwargs = copy.copy(kwargs)
                new_kwargs["crop"] = new_crop
                new_kwargs["saved"] = {}
                ZP = self.get_data("ZP", **new_kwargs)
                shape = ZP.shape
                ndim = ZP.ndim
                if ndim == 4 :
                    NT = shape[0]
                    ZP_ZSTAG = np.zeros((NT, NZ_ZSTAG, NY, NX))
                    ZP_ZSTAG[:, 0] = HT
                    for iz in range(1,NZ_ZSTAG):
                        ZP_ZSTAG[:, iz] = 2*ZP[:, iz-1] - ZP_ZSTAG[:, iz-1]
                    tab_slice = (slice(0, NT),)+tab_slice
                else :
                    ZP_ZSTAG = np.zeros((NZ_ZSTAG, NY, NX))
                    ZP_ZSTAG[0] = HT
                    for iz in range(1,NZ_ZSTAG):
                        ZP_ZSTAG[iz] = 2*ZP[iz-1] - ZP_ZSTAG[iz-1]
                return ZP_ZSTAG[tab_slice] 
            else :
                var_PHI = self.get_data("PHI_ZSTAG", **kwargs)
                return var_PHI/constants.G   
        elif varname in ["ZP_BASE"] :
            var_PHI = self.get_data("PHI_BASE", **kwargs)
            return var_PHI/constants.G  
        elif varname in ["Z", "Z_ZSTAG"] :
            var_ZP = self.get_data("ZP"+varname[1:], **kwargs)
            if varname.endswith("ZSTAG"):
                return var_ZP - self.get_data("HT3_ZSTAG", **kwargs)
            else :
                return var_ZP - self.get_data("HT3", **kwargs)    
        elif varname in ["DZ"] :
            new_kwargs = copy.copy(kwargs)
            if "crop" in kwargs :
                crop = kwargs["crop"]
                cropz = crop[0]
                if type(cropz) is int :
                    new_cropz = [cropz, cropz+2]
                elif type(cropz) is list :
                    new_cropz = [cropz[0], cropz[1]+1]
                elif cropz == "ALL" :
                    new_cropz = [0, self.get_data("NZ_ZSTAG")]
                else :
                    print(self.prefix, "warning : unexpected type of cropz in domWRF.calculate(DZ) : cropz = ", cropz, ", type(cropz) = ", type(cropz))
                    new_cropz = cropz
                new_crop = (new_cropz, crop[1], crop[2])
                new_kwargs["crop"] = new_crop
            ZP_ZSTAG = self.get_data("ZP_ZSTAG", **new_kwargs)
            TIME = self.get_data("TIME", **new_kwargs)
            if type(TIME) in [list, np.array, np.ndarray] and len(TIME) > 1 and len(ZP_ZSTAG) > 1:
                zaxis = 1
            else :
                zaxis = 0
            return np.diff(ZP_ZSTAG, axis=zaxis)
        elif varname in ["PID_BASE", "PID_BASE_ZSTAG"] : #See Skamarock 2019 eq. 2.2
            stagname = varname[8:]
            C3 = self.get_data("C3"+stagname, **kwargs) # c3 = b
            C4 = self.get_data("C4"+stagname, **kwargs) # c4 = (eta-b)*(p0-pt)
            P_TOP = self.get_data("P_TOP", **kwargs)[0]
            MUT_BASE = self.get_data("MUT_BASE", **kwargs)
            return C3*MUT_BASE + C4 + P_TOP
        elif varname in ["PID_PERT", "PID_PERT_ZSTAG"] : #See Skamarock 2019 eq. 2.2
            stagname = varname[8:]
            C3 = self.get_data("C3"+stagname, **kwargs) # c3 = b
            MUT_PERT = self.get_data("MUT_PERT", **kwargs)
            return C3*MUT_PERT
        elif varname in ["PNH", "PNH_AVG"] : # Non-hydrostatic pressure
            P, PI = self.get_data(["P_PERT_AVG", "PI_PERT_AVG"], **kwargs)
            return P-PI
        elif varname in ["MUD_BASE", "MUD_BASE_ZSTAG"] : #See Skamarock 2019 eq. 2.6
            stagname = varname[8:]
            C1 = self.get_data("C1"+stagname, **kwargs) #c1 = d b / d eta
            C2 = self.get_data("C2"+stagname, **kwargs) #c2 = (1-c1)*(p0-pt)
            MUT_BASE = self.get_data("MUT_BASE", **kwargs)
            C1 = np.expand_dims(C1, axis=(-2, -1))
            C2 = np.expand_dims(C2, axis=(-2, -1))
            return C1*MUT_BASE + C2
        elif varname in ["MUD_PERT", "MUD_PERT_ZSTAG"] : #See Skamarock 2019 eq. 2.6
            stagname = varname[8:]
            C1 = self.get_data("C1"+stagname, **kwargs) #c1 = d b / d eta
            C1 = np.expand_dims(C1, axis=(-2, -1))
            MUT_PERT = self.get_data("MUT_PERT", **kwargs)
            return C1*MUT_PERT
        elif varname == "VEG_FRAC":
            return self.get_data("VEGFRA", **kwargs)/100.0
        elif varname == "BOWEN":
            SH_FLX = self.get_data("SH_FLX", **kwargs)
            LH_FLX = self.get_data("LH_FLX", **kwargs)
            return SH_FLX/LH_FLX
        elif varname in ["TKE"] :
            return self.get_data("TKE_SFS", **kwargs) + self.get_data("TKE_RES", **kwargs)
        elif varname in ["TKE_RES"] :
            BL_PBL_PHYSICS = self.get_data("BL_PBL_PHYSICS")
            if BL_PBL_PHYSICS == 0 : #LES
                M2U = self.get_data("M2U", **kwargs)
                M2V = self.get_data("M2V", **kwargs)
                M2W = self.get_data("M2W", **kwargs)
                TKE_RES = 0.5 * (M2U + M2V + M2W)
            else :
                TKE_RES = self.get_data("ZERO", **kwargs)
            return TKE_RES
        elif varname == "NBV2" :
            #see Skamarock 2019, p.37 (section 4.2.4 Buoyancy) 
            #The calculation of NBV2 here is slightly different than (g/PTV)(dPTV/dZ) in but it is very close under the assumption that
            # 1) q=qv
            #A second assumption is made here :
            # 2) air is not saturated
            # PT = self.get_data("PT", **kwargs)
            # DZ_PT = self.get_data("DZ_PT", **kwargs)
            # DZ_QV = self.get_data("DZ_QV", **kwargs)
            # DZ_Q = self.get_data("DZ_Q", **kwargs)
            # return constants.G * (DZ_PT/PT + (1+constants.EPSILON) * DZ_QV - DZ_Q)
            
            # It has been checked that the result is similar with :
            # PTV = np.squeeze(self.get_data("PTV", **kwargs))
            # DZ_PTV = np.squeeze(self.get_data("DZ_PTV", **kwargs))
            PTV = self.get_data("PTV", **kwargs)
            DZ_PTV = self.get_data("DZ_PTV", **kwargs)
            return constants.G/PTV * DZ_PTV
        elif varname in ["KMH"]:
            KM_OPT = self.get_data("KM_OPT")
            if KM_OPT == 4 : # 2D smag
                LH2 = self.get_data("DX")*self.get_data("DY")
                CS2 = constants.CS**2
                D11 = self.get_data("D11", **kwargs)
                D22 = self.get_data("D22", **kwargs)
                D12 = self.get_data("D12", **kwargs)
                return CS2*LH2*np.sqrt(0.25*(D11-D22)**2 + D12**2)
            elif KM_OPT == 2 : # 3D TKE see Skamarock 2019 eq. 4.6 to 4.7
                LTURBH = self.get_data("LTURBH", **kwargs)
                TKE_SFS = self.get_data("TKE_SFS", **kwargs)
                return constants.CK*LTURBH*np.sqrt(TKE_SFS)
            else :
                raise(Exception("error in DomWRF.calculate(KMH, ...) cannot compute with KM_OPT = " + str(KM_OPT)))
        elif varname in ["KMV"]:
            BL_PBL_PHYSICS = self.get_data("BL_PBL_PHYSICS")
            if BL_PBL_PHYSICS == 2 : #MYJ #see Janjic (2002), Mellor and Yamada (1982)
                A1, A2, B1, B2, C1, BETA = constants.A1_MYJ, constants.A2_MYJ, constants.B1_MYJ, constants.B2_MYJ, constants.C1_MYJ, constants.BETA_MYJ
                TKE = self.get_data("TKE_PBL", **kwargs) #TKE_PBL output from the model
                DZ_U = self.get_data("DZ_U", **kwargs)
                DZ_V = self.get_data("DZ_V", **kwargs)
                DZ_PTV = self.get_data("DZ_PTV", **kwargs)
                LTURBV = self.get_data("LTURBV", **kwargs)
                Q = np.sqrt(2*TKE)
                GM = (LTURBV**2/Q**2) * (DZ_U**2 + DZ_V**2)
                GH = (LTURBV**2/Q**2) * BETA*constants.G * DZ_PTV
                # K1*SM + K2*SH = K3 (1)
                # K4*SM + K5*SH = K6 (2)
                K1 = 6*A1*A2*GM
                K2 = 1 - 3*A2*B2*GH - 12*A1*A2*GH
                K3 = A2
                K4 = 1 + 6*A1*A1*GM - 9*A1*A2
                K5 = 12*A1*A1*GH + 9*A1*A2*GH
                K6 = A1*(1-3*C1)
                #from (1) we get :
                # SH = (K3 - K1*SH) / K2
                #Injecting in (2) :
                # K4*SM + K5*K3/K2 - K5*K1*SM/K2 = K6
                #Then :
                SM = (K2*K6 - K5*K3)/(K2 - K5*K1)
                return LTURBV*Q*SM
            elif BL_PBL_PHYSICS == 0 : #LES
                KM_OPT = self.get_data("KM_OPT")
                if KM_OPT == 2 : # 3D Prognostic TKE non isotropic (see Skamarock 2019 after eq.4.6)
                    TKE_SFS = self.get_data("TKE_SFS", **kwargs)
                    LTURBV = self.get_data("LTURBV", **kwargs)
                    return constants.CK*LTURBV*np.sqrt(TKE_SFS)
                else : 
                    raise(Exception("error in DomWRF.calculate(KMV, ...) cannot compute with KM_OPT = " + str(KM_OPT)))
            else :
                raise(Exception("error in DomWRF.calculate(KMV, ...) cannot compute with BL_PBL_PHYSICS = " + str(BL_PBL_PHYSICS)))
        elif varname in ["LTURBV"]:
            BL_PBL_PHYSICS = self.get_data("BL_PBL_PHYSICS")
            if BL_PBL_PHYSICS == 2 : #MYJ #see Janjic (2002), Mellor and Yamada (1982)
                return self.get_data("EL_PBL", **kwargs) #PBL length scale output from the model
            elif BL_PBL_PHYSICS == 0 : #LES
                KM_OPT = self.get_data("KM_OPT")
                if KM_OPT == 2 : # 3D Prognostic TKE non isotropic (see Skamarock 2019 after eq.4.6)
                    if not "MIX_ISOTROPIC" in self.VARIABLES :
                        print(self.prefix, "warning : assuming non isotropic to calculate KMV in DomWRF.calculate for domain", self.name)
                        MIX_ISOTROPIC = False
                    else : 
                        MIX_ISOTROPIC = self.get_data("MIX_ISOTROPIC")
                    DZ = self.get_data("DZ", **kwargs)
                    if not MIX_ISOTROPIC : #anisotropic
                        LTURBV = DZ
                    else : #isotropic 
                        DX = self.get_data("DX")
                        DY = self.get_data("DY")
                        LTURBV = (DX*DY*DZ)**(1/3)
                    NBV2 = self.get_data("NBV2", **kwargs)
                    TKE_SFS = self.get_data("TKE_SFS", **kwargs)
                    temp = np.abs(NBV2)
                    temp[temp < 1e-10] = 1e-10
                    LTURBV_bis = 0.76*np.sqrt(TKE_SFS/temp)
                    pos = np.logical_and(NBV2 > 1e-9, LTURBV_bis < LTURBV)
                    LTURBV[pos] = LTURBV_bis[pos]
                    return LTURBV
                else : 
                    raise(Exception("error in DomWRF.calculate(LTURBV, ...) cannot compute with KM_OPT = " + str(KM_OPT)))
            else :
                raise(Exception("error in DomWRF.calculate(LTURBV, ...) cannot compute with BL_PBL_PHYSICS =" + str(BL_PBL_PHYSICS)))
        elif varname in ["LTURBH"]:
            KM_OPT = self.get_data("KM_OPT")
            if KM_OPT == 4 : # 2D smag
                return np.sqrt(self.get_data("DX")*self.get_data("DY"))
            elif KM_OPT == 2 : # 3D TKE see Skamarock 2019 eq. 4.6 to 4.7
                if not "MIX_ISOTROPIC" in self.VARIABLES :
                    print(self.prefix, "warning : assuming non isotropic to calculate KMH in DomWRF.calculate for domain", self.name)
                    MIX_ISOTROPIC = False
                else : 
                    MIX_ISOTROPIC = self.get_data("MIX_ISOTROPIC")
                DX = self.get_data("DX")
                DY = self.get_data("DY")
                if not MIX_ISOTROPIC : #anisotropic
                    LH = np.sqrt(DX*DY)
                else : #isotropic
                    TKE_SFS = self.get_data("TKE_SFS", **kwargs)
                    DZ = self.get_data("DZ", **kwargs)
                    LH = (DX*DY*DZ)**(1/3)
                    NBV2 = self.get_data("NBV2", **kwargs)
                    temp = np.abs(NBV2)
                    temp[temp < 1e-10] = 1e-10
                    LH_bis = 0.76*np.sqrt(TKE_SFS/temp)
                    pos = np.logical_and(NBV2 > 1e-10, LH_bis < LH)
                    LH[pos] = LH_bis[pos]
                return LH
            else :
                raise(Exception("error in DomWRF.calculate(LTURBH, ...) cannot compute with KM_OPT = " +str(KM_OPT)))
        elif varname in ["PRANDTLV"]:
            DZ = self.get_data("DZ", **kwargs)
            LTURBV = self.get_data("LTURBV", **kwargs)
            return 1/(1 + 2*LTURBV/DZ)
        elif varname in ["PRANDTLH"]:
            return 1/3
        elif varname in ["KQV"]:
            KMV = self.get_data("KMV", **kwargs)
            PRANDTLV = self.get_data("PRANDTLV", **kwargs)
            return KMV/PRANDTLV
        elif varname in ["KQH"]:
            KMH = self.get_data("KMH", **kwargs)
            PRANDTLH = self.get_data("PRANDTLH", **kwargs)
            return KMH/PRANDTLH
        elif varname in ["COVWPTV_SGS"] : #Turbulent thermal flux
            KQV = self.get_data("KQV", **kwargs)
            DZ_PTV = self.get_data("DZ_PTV", **kwargs)
            return -KQV*DZ_PTV
        elif varname.endswith("_TS") : #for example : "U_CRO_TS" 
            return self.calculate_ts(varname, **kwargs)
        elif varname.endswith("_TSAVG") or varname.endswith("_TSVAR")  or varname.endswith("_TSSTD") or varname.endswith("_TSCOV") or varname.endswith("_TSAV2") or varname.endswith("_TSM3"):
            return self.calculate_tsstats(varname, **kwargs)
        elif varname in self.avg_from_ac_list :
            time_slice = kwargs["time_slice"]
            return_zero = False
            insert_zero_before = False
            if type(time_slice) is int :
                if time_slice == 0 :
                    new_time_slice = slice(0, 2, 1)
                    return_zero = True
                else :
                    new_time_slice = slice(time_slice-1, time_slice+1, 1)
            elif type(time_slice) is slice :
                start = time_slice.start
                stop = time_slice.stop
                if start is None : 
                    start = 0
                if start > 0 :
                    new_time_slice = slice(start-1, stop, 1)
                else :
                    insert_zero_before = True
                    new_time_slice = time_slice
            elif type(time_slice) is list :
                if time_slice[0] == 0 :
                    insert_zero_before = True
                    new_time_slice = time_slice
                else :
                    new_time_slice = time_slice[0]-1
                    for it in time_slice :
                        new_time_slice.append(it)
            else :
                new_time_slice = time_slice
            new_kwargs = copy.copy(kwargs)
            new_kwargs["time_slice"] = new_time_slice
            print(self.prefix, new_time_slice)
            AC_var = self.get_data("AC"+varname[:-4], **new_kwargs)
            if return_zero :
                return 0*AC_var
            else :
                TIME = self.get_data("TIME", **new_kwargs)
                diff_AC_var = np.diff(AC_var, axis=0)
                diff_TIME_seconds = []
                for it in range(len(TIME) - 1):
                    diff_TIME_seconds.append((TIME[it+1] - TIME[it]).seconds)
                for idim in range(1, diff_AC_var.ndim):
                    diff_TIME_seconds = np.expand_dims(diff_TIME_seconds, -1)
                var_AVG = diff_AC_var/diff_TIME_seconds
                if insert_zero_before :
                    zero = np.zeros((1,)+var_AVG.shape[1:])
                    var_AVG = np.concatenate((zero, var_AVG), axis=0)
                return var_AVG
        else : #Try to look in Dom.calculate
            return Dom.calculate(self, varname, **kwargs)
    
    def calculate_statistics(self, varname, **kwargs):
        """
        Description
            Calculate the centered statistics based on the non-centered statistics calculated by WRFstats
        Parameters
            varname (str) : see Dom.get_data
        Optional
            kwargs : see Dom.get_data
        05/02/2025 : Mathieu LANDREAU
        """
        if varname in ["M2U", "M2V", "M2W", "M2Q", "M2QV", "M2RHO"] :
            varname_prefix = varname[2:]
            var_avg = self.get_data(varname_prefix+"_AVG", **kwargs)
            var2_avg = self.get_data(varname_prefix+"2_AVG", **kwargs)
            return var2_avg - var_avg**2 #<u'2> = <u2> - <u>2
        elif varname in ["M2PT", "M2PTV", "M2P", "M2ZP"] : #since centered variance is invariant in shift we use directly perturbations 
            varname_prefix = varname[2:]
            var_pert_avg = self.get_data(varname_prefix+"_PERT_AVG", **kwargs)
            var2_pert_avg = self.get_data(varname_prefix+"2_PERT_AVG", **kwargs)
            return var2_pert_avg - var_pert_avg**2
        elif varname.startswith("STD") :
            return np.sqrt(self.get_data("M2"+varname[3:], **kwargs))
        elif varname.startswith("COV") :
            if varname[3] in ["U", "V", "W"] : #COVUW, COVUPT, COVWT, 
                varname1 = varname[3]
                varname2 = varname[4:]
                if varname1 == varname2 :
                    return self.get_data("M2"+varname1, **kwargs)
                if varname2 in ["PT", "PTV", "T", "P", "ZP", "PI"] : 
                    varname2 = varname2 + "_PERT"
                product_avg = self.get_data(varname1+varname2+"_AVG", **kwargs)
                var1_avg = self.get_data(varname1+"_AVG", **kwargs)
                var2_avg = self.get_data(varname2+"_AVG", **kwargs)
                return product_avg - var1_avg*var2_avg  #<u'v'> = <uv> - <u><v>
        elif varname.startswith("M3") : #3rd order statistics (only for u, v, w)
            if varname[4:] in ["U", "V", "W"] :
                temp = [varname[2], varname[3], varname[4]]
                temp.sort() # M3UUW but not M3UWU
                v1, v2, v3 = temp
            else :
                temp = [varname[2], varname[3]]
                temp.sort() # M3UUW but not M3UWU
                v1, v2 = temp
                v3 = varname[4:]
            s123, s1, s2, s3 = self.get_data([v1+v2+v3+"_AVG", v1+"_AVG", v2+"_AVG", v3+"_AVG"], **kwargs)
            s12, s13, s23 = self.get_data([v1+"2_AVG" if v1==v2 else v1+v2+"_AVG", 
                                           v1+"2_AVG" if v1==v3 else v1+v3+"_AVG", 
                                           v2+"2_AVG" if v2==v3 else v2+v3+"_AVG"], **kwargs) 
            return s123 - s12*s3 - s13*s2 - s23*s1 + 2*s1*s2*s3
        else : #Try to look in Dom.calculate_statistics
            return Dom.calculate_statistics(self, varname, **kwargs)
    
    def calculate_stress_tensor_bilan_term(self, varname_in, p="", **kwargs) :
        """
        Description
            Calculate the terms of the stress tensor bilan (e.g. Deardorff 1980, https://doi.org/10.1007/BF00119502)
            Warning : The derivatives are in WRF coordinates (DXW_xx) because I calculated these terms over water where the correction from WRF to cartesian referential frame is negligible. Over land, it would be better to change the derivatives to DXC_xx
        Parameters
            varname : str : name of the variable 
            p : str : prefix for rotation (see Dom.calculate_rotation) : can be AD225_ or CC_, ... The rotation must be calculated on derivatives and variances first
        Optional
            kwargs : all the kwargs from get_data
        05/02/2025 : Mathieu LANDREAU
        """
        if varname_in.startswith("W") :
            W = "W" # calculate terms in WRF coordinates system (advection with omega, ...)
            varname = varname_in[1:]
            C = "W" # "W" = compute derivatives in WRF coordinate system
        else :
            W = ""
            varname = varname_in
            C = "C"  # "W" = compute derivatives in cartesian system
            
        # P: shear production, A: advection, T:pressure transport, D: turbulent diffusion, B: buoyancy, N: Non-hydrostatic pressure transport, R: "exact" pressure transport
        if varname in ["PTKE", "ATKE", "TTKE", "DTKE", "BTKE", "NTKE", "RTKE", "GTKE", "KTKE", "ZTKE"] : #total Production, Advection, pressure Transfer, turbulent Diffusion, Buoyancy, or Non-hydrostatic pressure
            vUU, vVV, vWW = self.get_data([p+W+varname[0]+"UU", p+W+varname[0]+"VV", p+W+varname[0]+"WW"], **kwargs) 
            return 0.5*(vUU+vVV+vWW)
        elif varname in ["PUU", "PVV", "PWW", "AUU", "AVV", "AWW", "AUV", "AUW", "AVW", "DUU", "DVV", "DWW", "DUV", "DUW", "DVW"] : #total production or advection or turbulent diffusion for 3 terms
            v1, v2, v3 = self.get_data([p+W+varname+"1", p+W+varname+"2", p+W+varname+"3"], **kwargs) 
            return v1 + v2 + v3
        elif varname in ["PUV", "PUW", "PVW"] : #total production for 6 terms
            v1, v2, v3, v4, v5, v6 = self.get_data([p+W+varname+"1", p+W+varname+"2", p+W+varname+"3", p+W+varname+"5", p+W+varname+"5", p+W+varname+"6"], **kwargs) 
            return v1 + v2 + v3 + v4 + v5 + v6
        
        # Shear production : P = - Rik.Uj,k - Rij.Ui,k
        ## Diagonal terms i=j
        elif varname in ["PUU1"] : #i=j=1, k=1
            M2U, DXW_U = self.get_data([p+"M2U", f"{p}DX{C}_{p}U"], **kwargs)
            return -2*M2U*DXW_U
        elif varname in ["PUU2"] : #i=j=1, k=2
            COVUV, DYW_U = self.get_data([p+"COVUV", f"{p}DY{C}_{p}U"], **kwargs)
            return -2*COVUV*DYW_U
        elif varname in ["PUU3"] : #i=j=1, k=3
            COVUW, DZ_U = self.get_data([p+"COVUW", f"DZ_{p}U"], **kwargs) if W == "" else self.get_data([p+"COVUO", f"DETA_{p}U"], **kwargs)
            return -2*COVUW*DZ_U
        elif varname in ["PVV1"] : #i=j=2, k=1
            COVUV, DXW_V = self.get_data([p+"COVUV", f"{p}DX{C}_{p}V"], **kwargs)
            return -2*COVUV*DXW_V
        elif varname in ["PVV2"] : #i=j=2, k=2
            M2V, DYW_V = self.get_data([p+"M2V", f"{p}DY{C}_{p}V"], **kwargs)
            return -2*M2V*DYW_V
        elif varname in ["PVV3"] : #i=j=2, k=3
            COVVW, DZ_V = self.get_data([p+"COVVW", f"DZ_{p}V"], **kwargs) if W == "" else self.get_data([p+"COVVO", f"DETA_{p}V"], **kwargs)
            return -2*COVVW*DZ_V
        elif varname in ["PWW1"] : #i=j=3, k=1
            COVUW, DXW_W = self.get_data([p+"COVUW", f"{p}DX{C}_W"], **kwargs)
            return -2*COVUW*DXW_W
        elif varname in ["PWW2"] : #i=j=3, k=2
            COVVW, DYW_W = self.get_data([p+"COVVW", f"{p}DY{C}_W"], **kwargs)
            return -2*COVVW*DYW_W
        elif varname in ["PWW3"] : #i=j=3, k=3
            M2W, DZ_W = self.get_data(["M2W", "DZ_W"], **kwargs) if W == "" else self.get_data(["COVWO", "DETA_W"], **kwargs)
            return -2*M2W*DZ_W
            
        ## Non-diagonal terms
        elif varname in ["PUV1"] :
            M2U, DXW_V = self.get_data([p+"M2U", f"{p}DX{C}_{p}V"], **kwargs)
            return -M2U*DXW_V
        elif varname in ["PUV2"] :
            COVUV, DYW_V = self.get_data([p+"COVUV", f"{p}DY{C}_{p}V"], **kwargs)
            return -COVUV*DYW_V
        elif varname in ["PUV3"] :
            COVUW, DZ_V = self.get_data([p+"COVUW", f"DZ_{p}V"], **kwargs)
            return -COVUW*DZ_V
        elif varname in ["PUV4"] :
            COVUV, DXW_U = self.get_data([p+"COVUV", f"{p}DX{C}_{p}U"], **kwargs)
            return -COVUV*DXW_U
        elif varname in ["PUV5"] :
            M2V, DYW_U = self.get_data([p+"M2V", f"{p}DY{C}_{p}U"], **kwargs)
            return -M2V*DYW_U
        elif varname in ["PUV6"] :
            COVVW, DZ_U = self.get_data([p+"COVVW", f"DZ_{p}U"], **kwargs)
            return -COVVW*DZ_U
        elif varname in ["PUW1"] :
            M2U, DXW_W = self.get_data([p+"M2U", f"{p}DX{C}_W"], **kwargs)
            return -M2U*DXW_W
        elif varname in ["PUW2"] :
            COVUV, DYW_W = self.get_data([p+"COVUV", f"{p}DY{C}_W"], **kwargs)
            return -COVUV*DYW_W
        elif varname in ["PUW3"] :
            COVUW, DZ_W = self.get_data([p+"COVUW", "DZ_W"], **kwargs)
            return -COVUW*DZ_W
        elif varname in ["PUW4"] :
            COVUW, DXW_U = self.get_data([p+"COVUW", f"{p}DX{C}_{p}U"], **kwargs)
            return -COVUW*DXW_U
        elif varname in ["PUW5"] :
            COVVW, DYW_U = self.get_data([p+"COVVW", f"{p}DY{C}_{p}U"], **kwargs)
            return -COVVW*DYW_U
        elif varname in ["PUW6"] :
            M2W, DZ_U = self.get_data(["M2W", f"DZ_{p}U"], **kwargs)
            return -M2W*DZ_U
        elif varname in ["PVW1"] :
            COVUV, DXW_W = self.get_data([p+"COVUV", f"{p}DX{C}_W"], **kwargs)
            return -COVUV*DXW_W
        elif varname in ["PVW2"] :
            M2V, DYW_W = self.get_data([p+"M2V", f"{p}DY{C}_W"], **kwargs)
            return -M2V*DYW_W
        elif varname in ["PVW3"] :
            COVVW, DZ_W = self.get_data([p+"COVVW", "DZ_W"], **kwargs)
            return -COVVW*DZ_W
        elif varname in ["PVW4"] :
            COVUW, DXW_V = self.get_data([p+"COVUW", f"{p}DX{C}_{p}V"], **kwargs)
            return -COVUW*DXW_V
        elif varname in ["PVW5"] :
            COVVW, DYW_V = self.get_data([p+"COVVW", f"{p}DY{C}_{p}V"], **kwargs)
            return -COVVW*DYW_V
        elif varname in ["PVW6"] :
            M2W, DZ_V = self.get_data(["M2W", f"DZ_{p}V"], **kwargs)
            return -M2W*DZ_V
        
        # Advection = Rij,k Uk
        elif varname in ["AUU1", "AVV1"] :
            U, DXW_M2U = self.get_data([p+"U", f"{p}DX{C}_{p}M2{varname[1]}"], **kwargs)
            return -U*DXW_M2U
        elif varname in ["AUU2", "AVV2"] :
            V, DYW_M2U = self.get_data([p+"V", f"{p}DY{C}_{p}M2{varname[1]}"], **kwargs)
            return -V*DYW_M2U
        elif varname in ["AUU3", "AVV3"] :
            W, DZ_M2U = self.get_data(["W", f"DZ_{p}M2{varname[1]}"], **kwargs) if W == "" else self.get_data(["O", f"DETA_{p}M2{varname[1]}"], **kwargs)
            return -W*DZ_M2U
        elif varname in ["AWW1"] :
            U, DXW_M2W = self.get_data([p+"U", f"{p}DX{C}_M2W"], **kwargs)
            return -U*DXW_M2W
        elif varname in ["AWW2"] :
            V, DYW_M2W = self.get_data([p+"V", f"{p}DY{C}_M2W"], **kwargs)
            return -V*DYW_M2W
        elif varname in ["AWW3"] :
            W, DZ_M2W = self.get_data(["W", "DZ_M2W"], **kwargs) if W == "" else self.get_data(["O", f"DETA_M2W"], **kwargs)
            return -W*DZ_M2W
        elif varname in ["AUV1", "AUW1", "AVW1"] :
            U, DXW_COV = self.get_data([p+"U", f"{p}DX{C}_{p}COV{varname[1:3]}"], **kwargs)
            return -U*DXW_COV
        elif varname in ["AUV2", "AUW2", "AVW2"] :
            V, DYW_COV = self.get_data([p+"V", f"{p}DY{C}_{p}COV{varname[1:3]}"], **kwargs)
            return -V*DYW_COV
        elif varname in ["AUV3", "AUW3", "AVW3"] :
            W, DZ_COV = self.get_data(["W", f"DZ_{p}COV{varname[1:3]}"], **kwargs) if W == "" else self.get_data(["O", f"DETA_{p}COV{varname[1:3]}"], **kwargs)
            return -W*DZ_COV
        
        # Boussinesq pressure transfer term
        elif varname in ["TUU"]:
            RHO, DXW_COVUP = self.get_data(["RHO", f"{p}DX{C}_{p}COVUP"], **kwargs)
            return -2*DXW_COVUP/RHO
        elif varname in ["TVV"]:
            RHO, DYW_COVVP = self.get_data(["RHO", f"{p}DY{C}_{p}COVVP"], **kwargs)
            return -2*DYW_COVVP/RHO
        elif varname in ["TWW"]:
            RHO, DZ_COVWP = self.get_data(["RHO", "DZ_COVWP"], **kwargs)
            return -2*DZ_COVWP/RHO
        elif varname in ["TUV"] :
            RHO, DXW_COVVP, DYW_COVUP = self.get_data(["RHO", f"{p}DX{C}_{p}COVVP", f"{p}DY{C}_{p}COVUP"], **kwargs)
            return -(DXW_COVVP + DYW_COVUP)/RHO
        elif varname in ["TUW"] :
            RHO, DXW_COVWP, DZ_COVUP = self.get_data(["RHO", f"{p}DX{C}_{p}COVWP", f"DZ_{p}COVUP"], **kwargs)
            return -(DXW_COVWP + DZ_COVUP)/RHO
        elif varname in ["TVW"] :
            RHO, DYW_COVWP, DZ_COVVP = self.get_data(["RHO", f"{p}DY{C}_COVWP", f"DZ_{p}COVVP"], **kwargs)
            return -(DYW_COVWP + DZ_COVVP)/RHO
        
        # Total pressure transfer term (including buoyancy)
        elif varname in ["RUU"]:
            return -2*self.get_data("COVUALPG1", **kwargs)
        elif varname in ["RVV"]:
            return -2*self.get_data("COVVALPG2", **kwargs)
        elif varname in ["RWW"]:
            return -2*self.get_data("COVWALPG3", **kwargs)
        
        # Estimation implicit diffusion
        elif varname in ["ZUU"]:
            return -(self.get_data("M2U", **kwargs)**1.5)/self.get_data("DX", **kwargs)
        elif varname in ["ZVV"]:
            return -(self.get_data("M2V", **kwargs)**1.5)/self.get_data("DY", **kwargs)
        elif varname in ["ZWW"]:
            return -(self.get_data("M2W", **kwargs)**1.5)/self.get_data("DZ", **kwargs)
        
        # Subgrid term 1
        elif varname in ["GUU"]:
            return 2*self.get_data("COVUS1", **kwargs)
        elif varname in ["GVV"]:
            return 2*self.get_data("COVVS2", **kwargs)
        elif varname in ["GWW"]:
            return 2*self.get_data("COVWS3", **kwargs)
        
        # Non-hydrostatic ressure transfer term
        elif varname in ["NUU"]:
            RHO, DXW_COVUP = self.get_data(["RHO", p+"DXW_"+p+"COVUPNH"], **kwargs)
            return -2*DXW_COVUP/RHO
        elif varname in ["NVV"]:
            RHO, DYW_COVVP = self.get_data(["RHO", p+"DYW_"+p+"COVVPNH"], **kwargs)
            return -2*DYW_COVVP/RHO
        elif varname in ["NWW"]:
            RHO, DZ_COVWP = self.get_data(["RHO", "DZ_COVWPNH"], **kwargs)
            return -2*DZ_COVWP/RHO
        elif varname in ["NUV"] :
            RHO, DXW_COVVP, DYW_COVUP = self.get_data(["RHO", p+"DXW_"+p+"COVVPNH", p+"DYW_"+p+"COVUPNH"], **kwargs)
            return -(DXW_COVVP + DYW_COVUP)/RHO
        elif varname in ["NUW"] :
            RHO, DXW_COVWP, DZ_COVUP = self.get_data(["RHO", p+"DXW_COVWPNH", "DZ_"+p+"COVUPNH"], **kwargs)
            return -(DXW_COVWP + DZ_COVUP)/RHO
        elif varname in ["NVW"] :
            RHO, DYW_COVWP, DZ_COVVP = self.get_data(["RHO", p+"DYW_COVWPNH", "DZ_"+p+"COVVPNH"], **kwargs)
            return -(DYW_COVWP + DZ_COVVP)/RHO
        
        # 2nd turbulent diffusion of TKE
        elif varname in ["KUU"]:
            return self.get_data("M3UUDIV", **kwargs)
        elif varname in ["KVV"]:
            return self.get_data("M3VVDIV", **kwargs)
        elif varname in ["KWW"]:
            return self.get_data("M3WWDIV", **kwargs)
        
        # Turbulent diffusion = <ui'uj'uk'>,k
        elif varname in ["DUU1", "DUU2", "DUU3", "DVV1", "DVV2", "DVV3", "DWW1", "DWW2", "DWW3"] :
            v1, v2 = varname[1], varname[2]
            i = int(varname[3])-1
            if i == 2 and W == "W" :
                i = 3
            deriv = [f"{p}DX{C}_", f"{p}DY{C}_", "DZ_", "DETA_"][i]
            if i < 3 :
                v3 = ["U", "V", "W"][i]
                temp = [v1, v2, v3]
                temp.sort() # M3UUW but not M3UWU
            else :
                temp = [v1, v2]
                temp.sort() # M3UUW but not M3UWU
                temp.append("O")
            M3 = "M3"+temp[0]+temp[1]+temp[2]
            if M3 not in ["M3WWW", "M3WWO"] :
                M3 = p+M3
            print(deriv+M3)
            return -self.get_data(deriv+M3, **kwargs)
        
        # Buoyancy : Bij = [gj <ui'theta_v'> + gi <uj' theta_v'>]/theta_v
        elif varname in ["BUW"] :
            PTV, COVUPTV = self.get_data(["PTV", p+"COVUPTV"], **kwargs)
            return constants.G/PTV * COVUPTV
        elif varname in ["BVW"] :
            PTV, COVVPTV = self.get_data(["PTV", p+"COVVPTV"], **kwargs)
            return constants.G/PTV * COVVPTV
        elif varname in ["BWW"] :
            PTV, COVWPTV = self.get_data(["PTV", "COVWPTV"], **kwargs)
            return 2*constants.G/PTV * COVWPTV
        elif varname in ["BUU", "BVV", "BUV"] :
            return self.get_data("ZERO", **kwargs)
        
        else : 
            raise(Exception(f"Unknown stress tensor bilan term : {varname}, check the is_stress_tensor_bilan_term function"))

    def calculate_stability(self, varname, **kwargs):
        """
        Description
            Calculate the stability variables specific to WRF
        Parameters
            varname : str : name of the variable 
        Optional
            kwargs : all the kwargs from get_data
        DD/MM/YYYY : Mathieu LANDREAU
        """
        if varname == "RIF" : #Flux Richardson 
            KM_OPT = self.get_data("KM_OPT")
            if KM_OPT == 4 : # RANS
                KH = self.get_data("KH", **kwargs)
                KM = self.get_data("KM", **kwargs)
                RI = self.get_data("RI", **kwargs)
                return KH*RI/KM
            elif KM_OPT == 2 : # LES, assuming we have statistics
                num = constants.G * self.get_data("COVWPTV", **kwargs) / self.get_data("PTV", **kwargs)
                # DXC_U = self.get_data("DXC_U", **kwargs)
                # DXC_V = self.get_data("DXC_V", **kwargs)
                # DXC_W = self.get_data("DXC_W", **kwargs)
                # DYC_U = self.get_data("DYC_U", **kwargs)
                # DYC_V = self.get_data("DYC_V", **kwargs)
                # DYC_W = self.get_data("DYC_W", **kwargs)
                DZ_U = self.get_data("DZ_U", **kwargs)
                DZ_V = self.get_data("DZ_V", **kwargs)
                # DZ_W = self.get_data("DZ_W", **kwargs)
                # M2U = self.get_data("M2U", **kwargs)
                # M2V = self.get_data("M2V", **kwargs)
                # M2W = self.get_data("M2W", **kwargs)
                # COVUV = self.get_data("COVUV", **kwargs)
                COVUW = self.get_data("COVUW", **kwargs)
                COVVW = self.get_data("COVVW", **kwargs)
                # den = DXC_U * M2U + DYC_V * M2V + DZ_W * M2W + (DXC_V + DYC_U) * COVUV + (DXC_W + DZ_U) * COVUW + (DYC_W + DZ_V) * COVVW
                den = DZ_U * COVUW + DZ_V * COVVW
                print(self.prefix, "RIF with LES stats")
                return num/den
        else : #Try to look in Dom.calculate
            return Dom.calculate_stability(self, varname, **kwargs)
    
    def calculate_derivative(self, varname, **kwargs):
        """:
        Calculate the spatial derivative of a value
        Note : for inherited class (like DomARPS, DomWRF, ...) the inherited method (DomWRF.calculate, ...) is called first and 
               if the variable isn't defined in the inherited method, then Dom.calculate is called
        varname : str : name of the variable 
        kwargs : all the kwargs from get_data
        27/02/2023 : Mathieu LANDREAU
        28/06/2024 : ML moved in class_domWRF
        """ 
        #ETA derivative in WRF referential frame (t, x, y, eta)
        if varname.startswith("DETA_") :
            varname2 = varname[5:]
            cropz, cropy, cropx = kwargs["crop"]
            cropz1, cropz2 = cropz
            if cropz1 == 0 or kwargs["quick_deriv"] :
                adjust_first = True
                new_cropz1 = cropz1
            else :
                adjust_first = False
                new_cropz1 = cropz1 - 1
            if cropz2 == self.get_data("NZ") or kwargs["quick_deriv"] :
                adjust_last = True
                new_cropz2 = cropz2
            else :
                adjust_last = False
                new_cropz2 = cropz2 + 1
            new_kwargs = copy.deepcopy(kwargs)
            new_crop = ([new_cropz1, new_cropz2], cropy, cropx)
            if debug : print(self.prefix, "new_crop : ", new_crop)
            new_kwargs["crop"] = new_crop
            new_kwargs["saved"] = {}
            ETA = self.get_data("ETA", **new_kwargs)
            var = self.get_data(varname2, **new_kwargs)
            ndim = var.ndim
            count = 0
            if ndim > 2 : #assume (Z, Y, X) or (t, Z, Y, X)
                zaxis = ndim - 3
            else : #assume (Z) or (t, Z)
                zaxis = ndim - 1 
            if ETA.ndim != ndim :
                idim = 0
                while idim < ndim :
                    len_idim = var.shape[idim]
                    if idim == ETA.ndim or ETA.shape[idim] != len_idim :
                        #ETA = np.tensordot(np.ones(len_idim), ETA, 0)
                        ETA = np.expand_dims(ETA, axis=idim)
                        ETA = np.concatenate((ETA,)*len_idim, axis=idim)
                    idim = idim + 1
            if debug : print(self.prefix, "zaxis : ", zaxis)
            if debug : print(self.prefix, varname2)
            if debug : print(self.prefix, "compare ETA shape : ", ETA.shape, var.shape)
            ETA_diff = np.diff(ETA, axis=zaxis)
            var_diff = np.diff(var, axis=zaxis)
            e1 = np.delete(ETA_diff, -1, axis=zaxis)
            e2 = np.delete(ETA_diff, 0, axis=zaxis)
            f1 = np.delete(var_diff, -1, axis=zaxis)
            f2 = np.delete(var_diff, 0, axis=zaxis)
            den = e1*e2*(e1 + e2)
            num = e2*e2*f1 + e1*e1*f2
            data = num/den
            if debug : print(self.prefix, "data.shape : ", data.shape)
            if adjust_first :
                NZ_temp = ETA_diff.shape[zaxis]
                ETA_diff1 = np.delete(ETA_diff, range(1,NZ_temp), axis=zaxis)
                var_diff1 = np.delete(var_diff, range(1,NZ_temp), axis=zaxis)
                if debug : print(self.prefix, "var_diff1.shape : ", var_diff1.shape)
                if debug : print(self.prefix, "ETA_diff1.shape : ", ETA_diff1.shape)
                if debug : print(self.prefix, "var_diff1/ETA_diff1.shape : ", (var_diff1/ETA_diff1).shape)
                #data = np.insert(data, 0, var_diff1/ETA_diff1, axis=zaxis)
                data = np.concatenate((var_diff1/ETA_diff1, data), axis=zaxis)
            if adjust_last :
                NZ_temp = ETA_diff.shape[zaxis]
                ETA_diffm1 = np.delete(ETA_diff, range(NZ_temp-1), axis=zaxis)
                var_diffm1 = np.delete(var_diff, range(NZ_temp-1), axis=zaxis)
                #data = np.insert(data, NZ_temp, var_diffm1/ETA_diffm1, axis=zaxis)
                data = np.concatenate((data, var_diffm1/ETA_diffm1), axis=zaxis)
            # data = np.squeeze(data)
            if debug : print(self.prefix, "data.shape : ", data.shape)
            return data
        
        #TIME derivative in cartesian referential frame (t, x, y, z)
        elif varname.startswith("DTC_") : 
            varname2 = varname[4:]
            DTW_var = self.get_data("DTW_"+varname2, **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DTW_var
            else : 
                DTW_ZP = self.get_data("DTW_ZP", **kwargs)
                # DETA_ZP = self.get_data("DETA_ZP", **kwargs)
                # DETA_var = self.get_data("DETA_"+varname2, **kwargs)
                # if debug : print(self.prefix, "DTC", DTW_var.shape, DTW_ZP.shape, DETA_var.shape, DETA_ZP.shape)
                # return DTW_var - DTW_ZP*DETA_var/DETA_ZP
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                if debug : print(self.prefix, "DTC", DTW_var.shape, DTW_ZP.shape, DZ_var.shape)
                return DTW_var - DTW_ZP*DZ_var
            
        #Z derivative in cartesian referential frame (t, x, y, z)
        # elif varname.startswith("DZ_") : 
        #     varname2 = varname[3:]
        #     DETA_ZP = self.get_data("DETA_ZP", **kwargs)
        #     DETA_var = self.get_data("DETA_"+varname2, **kwargs)
        #     if debug : print(self.prefix, "DZ", DETA_var.shape, DETA_ZP.shape)
        #     return DETA_var/DETA_ZP
        
        #Y derivative in cartesian referential frame (t, x, y, z)
        elif varname.startswith("DYC_") : 
            varname2 = varname[4:]
            DYW_var = self.get_data("DYW_"+varname2, **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DYW_var
            else : 
                DYW_ZP = self.get_data("DYW_ZP", **kwargs)
                # DETA_ZP = self.get_data("DETA_ZP", **kwargs)
                # DETA_var = self.get_data("DETA_"+varname2, **kwargs)
                # if debug : print(self.prefix, "DYC", DYW_var.shape, DYW_ZP.shape, DETA_var.shape, DETA_ZP.shape)
                # return DYW_var - DYW_ZP*DETA_var/DETA_ZP
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                if debug : print(self.prefix, "DYC", DYW_var.shape, DYW_ZP.shape, DZ_var.shape)
                return DYW_var - DYW_ZP*DZ_var
            
        #X derivative in cartesian referential frame (t, x, y, z)
        elif varname.startswith("DXC_") :
            varname2 = varname[4:]
            DXW_var = self.get_data("DXW_"+varname2, **kwargs)
            if self.get_dim(varname2) < 3 : 
                return DXW_var
            else : 
                DXW_ZP = self.get_data("DXW_ZP", **kwargs)
                # DETA_ZP = self.get_data("DETA_ZP", **kwargs)
                # DETA_var = self.get_data("DETA_"+varname2, **kwargs)
                # if debug : print(self.prefix, "DXC", DXW_var.shape, DXW_ZP.shape, DETA_var.shape, DETA_ZP.shape)
                # return DXW_var - DXW_ZP*DETA_var/DETA_ZP
                DZ_var = self.get_data("DZ_"+varname2, **kwargs)
                if debug : print(self.prefix, "DXC", DXW_var.shape, DXW_ZP.shape, DZ_var.shape)
                return DXW_var - DXW_ZP*DZ_var
        
        #call mother class Dom.calculate_derivative
        else : 
            return Dom.calculate_derivative(self, varname, **kwargs)
            
    def calculate_ts(self, varname, **kwargs):
        if varname[-7] == "_" : #loc is a 3 letters char
            varname_short = varname[:-7]
            loc = varname[-6:-3]
        elif varname[-6] == "_" : #loc is a 2 letters char
            varname_short = varname[:-6]
            loc = varname[-5:-3]
        elif varname[-5] == "_" : #loc is a 1 letters char
            varname_short = varname[:-5]
            loc = varname[-4:-3]

        if varname_short in self.dict_varname_ts :
            first_line, last_line = self.get_ts_time_slice(loc, **kwargs)
            _, var, _ = self.get_ts(loc, varname_short, first_line=first_line, last_line=last_line, **kwargs)
            return var
        elif varname_short == "TIME" :
            first_line, last_line = self.get_ts_time_slice(loc, **kwargs)
            TIME_TS, _, _ = self.get_ts(loc, "U10", first_line=first_line, last_line=last_line, **kwargs)
            return TIME_TS
            """
            elif varname_short == "ZP" : 
            ZPTS = self.get_data("ZPTS_" + loc + "_TS", **kwargs)
            HT = self.get_ts_params(loc)["HT"]
            print(self.prefix, ZPTS)
            print(self.prefix, ZPTS.shape)
            if ZPTS.ndim == 1 :
                ZP_ZSTAG = np.concatenate((np.array([HT]), ZPTS))
                ZP = (ZP_ZSTAG[1:] + ZP_ZSTAG[:-1])*0.5
            else :
                NT = len(ZPTS)
                print(self.prefix, ZPTS.shape)
                ZP_ZSTAG = np.concatenate((np.expand_dims(np.array(NT*[HT]), axis=-1), ZPTS), axis=-1)
                print(self.prefix, ZP_ZSTAG.shape)
                ZP = (ZP_ZSTAG[:, 1:] + ZP_ZSTAG[:, :-1])*0.5
            return ZP
            """
        elif varname_short == "MH" : 
            Uname = "U_" + loc + "_TS"
            Vname = "V_" + loc + "_TS"
            return np.sqrt(self.get_data(Uname, **kwargs)**2 + self.get_data(Vname, **kwargs)**2)
        elif varname_short == "M" : 
            Uname = "U_" + loc + "_TS"
            Vname = "V_" + loc + "_TS"
            Wname = "W_" + loc + "_TS"
            return np.sqrt(self.get_data(Uname, **kwargs)**2 + self.get_data(Vname, **kwargs)**2 + self.get_data(Wname, **kwargs)**2)
        elif varname_short == "PTV" : 
            PT = self.get_data("PT_" + loc + "_TS", **kwargs)
            QV = self.get_data("QV_" + loc + "_TS", **kwargs)
            return PT*(1+constants.RV/constants.RD*QV)
        elif varname_short == "RHOD" : 
            P = self.get_data("P_" + loc + "_TS", **kwargs)
            PTV = self.get_data("PTV_" + loc + "_TS", **kwargs)
            TV = constants.PT_to_T(PTV, P)
            return P/(constants.RD*TV)
        elif varname_short == "RHOV" : 
            RHOD = self.get_data("RHOD_" + loc + "_TS", **kwargs)
            QV = self.get_data("QV_" + loc + "_TS", **kwargs)
            #IMPORTANT NOTE : This formula is not complete but is close to the truth. We should add, QRAIN, QCLOUD, QICE, ...
            return RHOD*QV
        elif varname_short == "RHO" : 
            RHOD = self.get_data("RHOD_" + loc + "_TS", **kwargs)
            RHOV = self.get_data("RHOV_" + loc + "_TS", **kwargs)
            #IMPORTANT NOTE : This formula is not complete but is close to the truth. We should add, QRAIN, QCLOUD, QICE, ...
            return RHOD + RHOV
        elif varname_short == "Q" : 
            return self.get_data("QV_" + loc + "_TS", **kwargs)
            #IMPORTANT NOTE : This formula is not complete but is close to the truth. We should add, QRAIN, QCLOUD, QICE, ...
        elif varname_short == "T" : 
            PT = self.get_data("PT_" + loc + "_TS", **kwargs)
            P = self.get_data("P_" + loc + "_TS", **kwargs)
            return constants.PT_to_T(PT, P)
        else : 
            raise(Exception("error in WRF.calculate : unknow tslist variable : " + varname + ", " + varname_short))
    
    def calculate_tsstats(self, varname, **kwargs):
        # Get all info from varname
        temp = varname.split("_")
        typ = temp[-1][2:]
        pfx = temp[-2]
        varname1 = "_".join(temp[0:-2])
        varname2 = varname3 = None
        if typ in ["AVG", "VAR", "STD"]: #for example : "U_CRO_TSAVG"
            pass
        elif typ in ["COV", "AV2"] : #covariance (COV) and non-cenetered moments : "UV_CRO_TSCOV", #WPTV_CRO_TSCOV # velocities should be first
            varname2 = varname1[1:]
            varname1 = varname1[0]
        elif typ in ["M3"] : #3rd order centered moments : "UVV_CRO_TSM3",
            assert(len(varname1) == 3)
            temp = [varname1[0], varname1[1], varname1[2]]
            temp.sort()
            varname1, varname2, varname3 = temp
        else :
            raise(Exception("unknown type of statistics in domWRF.calculate_tsstats : " + str(typ)))
        # Manage time
        TIME_STATS = self.get_data("TIME_STATS", **kwargs)
        if type(TIME_STATS) in [list, np.array, np.ndarray] :
            TIME_STATS = list(TIME_STATS)
        else :
            TIME_STATS = [TIME_STATS]
        NT_STATS = len(TIME_STATS)+1
        #the first date is before TIME_STATS[0] because TIME_STATS[0] is the end of the first interval
        TIME_STATS = np.array([TIME_STATS[0] - self.get_data("DT_STATS")] + TIME_STATS)
        TIME_TS = self.get_data("TIME_"+pfx+"_TS", **kwargs)
        NT_TS = len(TIME_TS)
        it_ts = 0
        tab_it_ts = []
        for it_stats in range(NT_STATS) :
            date_i = TIME_STATS[it_stats]
            while it_ts < NT_TS and TIME_TS[it_ts] <= date_i :
                it_ts += 1
            tab_it_ts.append(it_ts)
        # Getting variables
        VAR_TS = self.get_data(varname1+"_"+pfx+"_TS", **kwargs)
        if varname2 is not None : VAR2_TS = self.get_data(varname2+"_"+pfx+"_TS", **kwargs)
        if varname3 is not None : VAR3_TS = self.get_data(varname3+"_"+pfx+"_TS", **kwargs)
        NZ = VAR_TS.shape[1]
        VAR_out = np.zeros((NT_STATS-1, NZ))
        AVG_it = np.zeros((NZ))
        for it_stats in range(NT_STATS -1) :
            it_ts1 = tab_it_ts[it_stats]
            it_ts2 = tab_it_ts[it_stats+1]
            nt_ts_i = it_ts2 - it_ts1
            AVG_it = np.mean(VAR_TS[it_ts1:it_ts2, :], axis=0)
            if varname2 is not None : AVG2_it = np.mean(VAR2_TS[it_ts1:it_ts2, :], axis=0)
            if varname3 is not None : AVG3_it = np.mean(VAR3_TS[it_ts1:it_ts2, :], axis=0)
            if typ == "AVG" :
                VAR_out[it_stats, :] = AVG_it
            elif typ == "VAR" : # here VAR means variance not variable
                AVG_it3 = np.tensordot(np.ones(nt_ts_i), AVG_it, 0)
                VAR_out[it_stats, :] = np.mean( (VAR_TS[it_ts1:it_ts2, :]-AVG_it3)**2 , axis=0)
            elif typ == "STD" : # here VAR means variance not variable
                AVG_it3 = np.tensordot(np.ones(nt_ts_i), AVG_it, 0)
                VAR_out[it_stats, :] = np.sqrt(np.mean( (VAR_TS[it_ts1:it_ts2, :]-AVG_it3)**2 , axis=0))
            elif typ == "COV":
                VAR_out[it_stats, :] = np.mean(VAR_TS[it_ts1:it_ts2, :]*VAR2_TS[it_ts1:it_ts2, :] , axis=0) - AVG_it*AVG2_it
            elif typ == "AV2": #UV_CRO_AV2 = <uv>
                VAR_out[it_stats, :] = np.mean(VAR_TS[it_ts1:it_ts2, :]*VAR2_TS[it_ts1:it_ts2, :] , axis=0)
            elif typ == "M3":
                t3 = np.mean(VAR_TS[it_ts1:it_ts2, :]*VAR2_TS[it_ts1:it_ts2, :]*VAR3_TS[it_ts1:it_ts2, :] , axis=0)
                t21 = np.mean(VAR_TS[it_ts1:it_ts2, :]*VAR2_TS[it_ts1:it_ts2, :], axis=0)*AVG3_it
                t22 = np.mean(VAR_TS[it_ts1:it_ts2, :]*VAR3_TS[it_ts1:it_ts2, :], axis=0)*AVG2_it
                t23 = np.mean(VAR2_TS[it_ts1:it_ts2, :]*VAR3_TS[it_ts1:it_ts2, :], axis=0)*AVG_it
                t1 = AVG_it*AVG2_it*AVG3_it
                VAR_out[it_stats, :] = t3 - t21 - t22 - t23 + 2*t1
            else :
                raise(Exception("unknown type of statistics in domWRF.calculate_tsstats : " + str(typ)))
        return np.squeeze(VAR_out)
    
    def get_ts_params(self, pfx) :
        """
        Read time series files if option is activated in WRF
        """
        if not pfx in self.output_filenames["ts"] :
            manage_dict.print_dict(self.output_filenames["ts"], "self.output_filenames['ts']")
            raise(Exception("error : the prefix (" + pfx + ") doesn't exist in self.output_filenames['ts']"))
        sfx = pfx+".d"+self.i_str+".UU"
        ts_filename = ""
        for filename in self.output_filenames["ts"][pfx] :
            if filename.endswith(sfx) :
                ts_filename = filename
        if ts_filename == "" :
            manage_dict.print_dict(self.output_filenames["ts"][pfx] , "self.output_filenames['ts']["+pfx+"]")
            raise(Exception("error : no file endswith (" + sfx + ") in self.output_filenames['ts']["+pfx+"]"))
        with open(ts_filename) as ts_file :
            header = ts_file.readline()
        # I hope that the header format is really fixed
        ts_params = {
            "name"  : header[:24],
            "dom"   : int(header[25:28]),
            "i_ts"  : int(header[28:32]),
            "pfx"   : header[32:36],
            "plat"  : float(header[39:46]),
            "plon"  : float(header[48:55]),
            "iy"    : int(header[58:62]) - 1, #convert to python indices
            "ix"    : int(header[63:67]) - 1, #convert to python indices
            "LAT"   : float(header[70:77]),
            "LON"   : float(header[78:86]),
            "HT"    : float(header[87:95]),
            "init"  : manage_time.to_datetime(header[-20:-1]),
        }
        return ts_params
    
    def get_ts_time_slice(self, loc, **kwargs):
        TIME_TS, _, _ = self.get_ts(loc, "U10", **kwargs)
        TIME = self.get_data("TIME", **kwargs)
        if type(TIME) in [list, np.array, np.ndarray] :
            TIME = list(TIME)
        else :
            TIME = [TIME]
        last_line = np.sum(np.array(TIME_TS) <= TIME[-1])
        first_line = len(TIME_TS) - np.sum(np.array(TIME_TS) >= TIME[0] - self.get_data("DT_STATS")) #substract 1 history time_step for ts averaging
        return first_line, last_line
    
    def get_ts(self, pfx, varname, first_line=0, last_line=None, crop=None, **kwargs):
        """
        Read time series files if option is activated in WRF
        """
        if not pfx in self.output_filenames["ts"] :
            manage_dict.print_dict(self.output_filenames["ts"], "self.output_filenames['ts']")
            raise(Exception("error : the prefix (" + pfx + ") doesn't exist in self.output_filenames['ts']"))
        if not varname in self.dict_varname_ts :
            manage_dict.print_dict(self.dict_varname_ts, "self.dict_varname_ts")
            raise(Exception("error : the variable (" + varname + ") doesn't exist in self.dict_varname_ts"))
        index = None
        varname_ts = self.dict_varname_ts[varname]
        if type(varname_ts) is list :
            index = varname_ts[1]
            varname_ts = varname_ts[0]
        sfx = pfx+".d"+self.i_str+"."+varname_ts
        ts_filename = ""
        for filename in self.output_filenames["ts"][pfx] :
            if filename.endswith(sfx) :
                ts_filename = filename
        if ts_filename == "" :
            manage_dict.print_dict(self.output_filenames["ts"][pfx] , "self.output_filenames['ts']["+pfx+"]")
            raise(Exception("error : no file endswith (" + sfx + ") in self.output_filenames['ts']["+pfx+"]"))
        with open(ts_filename) as ts_file :
            header = ts_file.readline()
            ts = np.genfromtxt(ts_file, max_rows=last_line)
            if last_line == 1 :
                ts = np.expand_dims(ts, axis=0)
            if index is None :
                t = ts[:, 0]
                var = ts[:, 1+crop[0][0]:1+crop[0][1]]
            else :
                t = ts[:, 1] #in pfx.dXX.TS file, time is the 2nd column
                var = ts[:, index]
            t = t[first_line:]
            var = var[first_line:]
        delta = manage_time.to_timedelta((t*3600*1e6).astype('timedelta64[ms]'))
        SIM_INITIAL_TIME = self.get_data("SIM_INITIAL_TIME")
        correct_with_DT = True
        DT = self.get_data("DT")
        if type(delta) in [np.array, list, np.ndarray] :
            date = []
            for d in delta :
                if correct_with_DT : 
                    d = round(d/DT)*DT
                date.append(SIM_INITIAL_TIME + d)
            date = np.array(date)
        else :
            if correct_with_DT : 
                date = SIM_INITIAL_TIME + delta
            else :
                date = delta
        return date, var, header
    
    def prepare_crop_unstag(self, crop, count_dim, tab_slice, i_unstag, old_shape) :
        """
        crop can be
            int, np.int64 : index in the axis
            list : first and last index in the axis
            "ALL"
        """
        error_dims = False
        if type(crop) is tuple and len(crop) != len(old_shape) :
            if len(old_shape) == 4 : #Assuming it is (?, z, y, x)
                crop_temp = crop
                if len(crop_temp) > 4 :
                    crop = (crop_temp[-4], crop_temp[-3], crop_temp[-2], crop_temp[-1])
                elif len(crop_temp) == 3:
                    crop = ("ALL", crop_temp[-3], crop_temp[-2], crop_temp[-1])
                else :
                    error_dims = True
                
            elif len(old_shape) == 3 : #Assuming it is (z, y, x)
                crop_temp = crop
                if len(crop_temp) > 3 :
                    crop = (crop_temp[-3], crop_temp[-2], crop_temp[-1])
                else :
                    error_dims = True
            elif len(old_shape) == 2 : #Assuming it is (y, x)
                crop_temp = crop
                crop = (crop_temp[-2], crop_temp[-1])
            elif len(old_shape) == 1 : #Assuming it is (z)
                crop_temp = crop
                crop = (crop_temp[-3],)
            else :
                error_dims = True
                
            if error_dims :
                raise(Exception("error in Variable.prepare_crop_unstag : crop = " + str(crop) + ", while old_shape = " + str(old_shape)))
            
        shape = ()         
        squeeze_dim_unstag = False
        dim_unstag = None
        if crop is not None :
            for itemp, temp in enumerate(crop) :
                if itemp == i_unstag :
                    dim_unstag = count_dim
                    if type(temp) in [int, np.int64] :
                        shape += (2,)
                        tab_slice += (slice(temp,temp+2),)
                        count_dim+=1
                        squeeze_dim_unstag = True
                    elif type(temp) is list :
                        shape += (temp[1]+1 - temp[0],)
                        tab_slice += (slice(temp[0], temp[1]+1),)
                        count_dim+=1
                    elif type(temp) is str and temp == "ALL" :
                        shape += (old_shape[itemp],)
                        tab_slice += (slice(old_shape[itemp]+1),)
                        count_dim+=1
                    else :
                        raise(Exception("error in VariableRead.get_data : crop must be a list of string or a list of list, temp = " + str(temp) + ", type(temp) = " + str(type(temp)) + ", crop = " + str(crop)))
                
                else :
                    if type(temp) in [int, np.int64]:
                        #shape lose a dimension
                        tab_slice += (temp,)
                    elif type(temp) is list :
                        shape += (temp[1] - temp[0],)
                        tab_slice += (slice(temp[0], temp[1]),)
                        count_dim+=1
                    elif type(temp) is str and temp == "ALL" :
                        shape += (old_shape[itemp],)
                        tab_slice += (slice(old_shape[itemp]+1),)
                        count_dim+=1
                    else :
                        raise(Exception("error in VariableRead.get_data : crop must be a list of string or a list of list, temp = " + str(temp) + ", type(temp) = " + str(type(temp)) + ", crop = " + str(crop)))
        else :
            for itemp, temp in enumerate(old_shape) :
                shape += (temp,)
                tab_slice += (slice(temp+1),)
                if itemp == i_unstag :
                    dim_unstag = count_dim
                count_dim+=1
        return tab_slice, dim_unstag, squeeze_dim_unstag, shape