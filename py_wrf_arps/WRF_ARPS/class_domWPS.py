#!/usr/bin/env python3
from .class_domWRF import DomWRF
from .class_dom import Dom
from ..class_variables import *
from ..lib import manage_projection, manage_list, constants

import os
import numpy as np
import copy
from netCDF4 import Dataset
from wrf import vertcross, CoordPair, interplevel, WrfProj

class DomWPS(DomWRF):
    software = "WPS"
    suffix_length = 22 #2020-05-14_00:00:00.nc
    
    def __init__(self, *args, **kwargs):
        """
        Extract data for a domain WRF or ARPS

        Parameters
        ----------
        self: DomWPS
        proj_dir: str #"scratch/data-mod/mlandreau/03_simulation/"
            Global path toward the whole simulation
        data_dir : str #"02_20200519/"
            name of the simulaiton directory
        tab_dom_str : list of str #["05", "04", "03"]
            List of domaine names. First elements of the list is the path 
            for the data of the smallest domain (if nested simulations).
            

        Optional
        ----------

        Returns
        ----------

        Author(s)
        ----------
        06/01/2023 : Mathieu Landreau
        """  
        super().__init__(*args, **kwargs)
    
    #def get_output_filenames : inherit from domWRF
        
    def init_saved_variables(self):
        filename = next(iter(self.output_filenames['hist']))
        ncfile = self.output_filenames['hist'][filename]
        
        #Getting terrain height nad landmask
        self.VARIABLES['HT'].value = HT = ncfile['HGT_M'][:][0]
        self.VARIABLES['LANDMASK'].value = ncfile['LANDMASK'][:][0]

        #Getting simulation initial time
        self.VARIABLES['INITIAL_TIME'].value = self.str2date(ncfile.SIMULATION_START_DATE,"%Y-%m-%d_%H:%M:%S")

        #Getting projection and central coordinates
        self.VARIABLES['MAPPROJ'].value = MAPPROJ = ncfile.MAP_PROJ
        self.VARIABLES['CTRLAT'].value = CTRLAT = ncfile.CEN_LAT
        self.VARIABLES['CTRLON'].value = CTRLON = ncfile.CEN_LON

        #Getting dimensions
        self.VARIABLES['NX'].value  = NX = ncfile.dimensions["west_east"].size
        self.VARIABLES['NY'].value  = NY = ncfile.dimensions["south_north"].size
        self.VARIABLES['NZ'].value  = NZ = ncfile.dimensions["num_metgrid_levels"].size
        self.VARIABLES['NX_XSTAG'].value  = NX_XSTAG = ncfile.dimensions["west_east_stag"].size
        self.VARIABLES['NY_YSTAG'].value  = NY_YSTAG = ncfile.dimensions["south_north_stag"].size
        self.VARIABLES['NZ_ZSTAG'].value  = NZ_ZSTAG = NZ + 1
        self.VARIABLES['NZ_SOIL'].value  = NZ_SOIL = ncfile.dimensions["num_st_layers"].size
        self.VARIABLES['NZ_SOIL_ZSTAG'].value  = NZ_SOIL_ZSTAG = ncfile.dimensions["num_st_layers"].size

        #Getting Latitude and Longitude
        self.VARIABLES['LAT'].value = LAT = ncfile['XLAT_M'][:][0]
        self.VARIABLES['LON'].value = LON = ncfile['XLONG_M'][:][0]  
        self.VARIABLES['LAT_XSTAG'].value = LAT_XSTAG = ncfile['XLAT_U'][:][0]
        self.VARIABLES['LON_XSTAG'].value = LON_XSTAG = ncfile['XLONG_U'][:][0]
        self.VARIABLES['LAT_YSTAG'].value = LAT_YSTAG = ncfile['XLAT_V'][:][0]
        self.VARIABLES['LON_YSTAG'].value = LON_YSTAG = ncfile['XLONG_V'][:][0]

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
        self.VARIABLES['DX'].value = ncfile.DX
        self.VARIABLES['DY'].value = ncfile.DY
        self.VARIABLES['DX_PROJ'].value = ncfile.DX
        self.VARIABLES['DY_PROJ'].value = ncfile.DY

        #Getting Z mesh
        Z_SOIL_ZSTAG_un = np.array([0, 0.07, 0.28, 1, 2.89])
        Z_SOIL_un = manage_list.unstag(Z_SOIL_ZSTAG_un, 0)

        Z_SOIL = np.zeros((NZ_SOIL, NY, NX))
        ZP_SOIL = np.zeros((NZ_SOIL, NY, NX))
        Z_SOIL_ZSTAG = np.zeros((NZ_SOIL_ZSTAG, NY, NX))
        ZP_SOIL_ZSTAG = np.zeros((NZ_SOIL_ZSTAG, NY, NX))
        for k in range(NZ_SOIL):
            Z_SOIL[k, :, :] = Z_SOIL_un[k]
            ZP_SOIL[k, :, :] = Z_SOIL[k, :, :] + HT
        for k in range(NZ_SOIL_ZSTAG):
            Z_SOIL_ZSTAG[k, :, :] = Z_SOIL_ZSTAG_un[k]
            ZP_SOIL_ZSTAG[k, :, :] = Z_SOIL_ZSTAG[k, :, :] + HT

        Z = ncfile['GHT'][:][0]

        self.VARIABLES['Z_SOIL'].value = Z_SOIL
        self.VARIABLES['ZP_SOIL'].value = ZP_SOIL
        self.VARIABLES['Z_SOIL_ZSTAG'].value = Z_SOIL_ZSTAG
        self.VARIABLES['ZP_SOIL_ZSTAG'].value = ZP_SOIL_ZSTAG
        self.VARIABLES['DZ_SOIL'].value = np.diff(Z_SOIL_ZSTAG_un)
        self.VARIABLES['ZP'].value = ZP = ncfile['GHT'][:][0]
        HT3 = np.tensordot(np.ones(NZ), HT, 0)
        self.VARIABLES["HT3"] = VariableSave("", "", "", "", 0, HT3) 
        self.VARIABLES['Z'].value = ZP - HT3
        self.VARIABLES['ZEROS'] = VariableSave("", "", "", "", 0, np.zeros((NZ, NY, NX)))
        self.VARIABLES['ZEROS_XSTAG'] = VariableSave("", "", "", "", 0, np.zeros((NZ, NY, NX_XSTAG)))
        self.VARIABLES['ZEROS_YSTAG'] = VariableSave("", "", "", "", 0, np.zeros((NZ, NY_YSTAG, NX)))
        self.VARIABLES['ZEROS_ZSTAG'] = VariableSave("", "", "", "", 0, np.zeros((NZ_ZSTAG, NY, NX)))
        self.VARIABLES['NMK'].value = NMK = 0
    
    def calculate(self, varname, **kwargs):
        if varname in ["T_PERT", "PT_PERT", "P_PERT", "U_PERT", "V_PERT", "W_PERT", "M_PERT", "MH_PERT", "WD_PERT"]: #BASE = 0 => PERT = Total
            return self.get_data(varname[:-5], **kwargs_temp)
        elif varname in ["U_PERT_XSTAG", "V_PERT_YSTAG"]:#BASE = 0 => PERT = Total
            return self.get_data(varname[:-11] + varname[-6:], **kwargs)
        elif varname in ["U_BASE", "V_BASE", "W_BASE", "M_BASE", "MH_BASE", "WD_BASE", "W", "W_PERT", "W_BASE", "PT_BASE"]: #BASE = 0
            return self.get_data("ZERO", **kwargs)
        elif varname in ["U_BASE_XSTAG", "V_BASE_YSTAG","W_BASE_ZSTAG", "W_ZSTAG", "W_PERT_ZSTAG"]: #BASE = 0
            return self.get_data("ZERO"+varname[-6:], **kwargs)
        
        elif varname in ["U"]:
            kwargs_temp = copy.copy(kwargs)
            kwargs_temp["i_unstag"] = 2
            return self.get_data(varname+"_XSTAG", **kwargs_temp)
        elif varname in ["V"]:
            kwargs_temp = copy.copy(kwargs)
            kwargs_temp["i_unstag"] = 1 
            return self.get_data(varname+"_YSTAG", **kwargs_temp)
        elif varname in ["PT"]:
            var_T = self.get_data(varname[1:], **kwargs)
            P = self.get_data("P", **kwargs)
            return constants.T_to_PT(var_T, P)
        elif varname == "T_SOIL":
            #WARNING, if z dimension is cropped, NZ_SOIL will be wrong
            # DO NOT USE WITH Z CROP, ALWAYS USE WITH NO CROP OR WITH CROP=("ALL", .., ..)
            LANDMASK = self.get_data("LANDMASK")
            NZ_SOIL = self.get_data("NZ_SOIL")
            SST = self.get_data("SST")
            T_SOIL = self.get_data("ST", **kwargs)[::-1]
            pos = np.where(LANDMASK == 0)
            for k in range(NZ_SOIL): #Same as WRF2ARPS
                T_SOIL[k][pos] = SST[pos]
            return T_SOIL 
        elif varname == "Q_SOIL":
            return self.get_data("SM", **kwargs)[::-1]
        else :
            return Dom.calculate(self, varname, **kwargs)

        