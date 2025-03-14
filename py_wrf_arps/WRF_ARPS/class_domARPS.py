#!/usr/bin/env python3
import sys

from .class_dom import Dom
from ..class_variables import VariableSave, Variable
from ..lib import constants, manage_time, manage_list, manage_projection

import os
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
from wrf import vertcross, CoordPair, interplevel, WrfProj
import collections #to sort dictionnaries
import copy #to modify kwargs to get moments

""" 
--- Note Mathieu Landreau 2024 
BE CAREFUL : 
Many things have been implemented in class_Proj, class_Dom, ... without being tested with a DomArps object
I am not using ARPS anymore
"""


class DomARPS(Dom):
    """
    Be careful : lon = x, lat = y
    """
    software = "ARPS"
    suffix_length = 6 #001800 for example

    def __init__(self, *args, **kwargs):
        """
        See class_Dom.__init__
        06/01/2023 : Mathieu Landreau
        """  
        super().__init__(*args, **kwargs)
    
    def get_output_filenames(self):
        self.output_filenames = {
            "hist" : {},
            "soil" : {},
            "stats" : {},
            "base" : {},
            "trn" : {},
            "sfc" : {},
            "post" : {},
        }
                     
        for filename in os.listdir(self.output_data_dir):
            if filename.endswith(self.output_prefix + '.netgrdbas'):
                if self.keep_open :
                    self.output_filenames['base'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['base'][self.output_data_dir + filename] = True
                self.FLAGS['base'] = True
            elif filename.endswith(self.output_prefix + '.trndata'):
                if self.keep_open :
                    self.output_filenames['trn'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['trn'][self.output_data_dir + filename] = True
                self.FLAGS['trn'] = True
            elif filename.endswith(self.output_prefix + '.sfcdata'):
                if self.keep_open :
                    self.output_filenames['sfc'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['sfc'][self.output_data_dir + filename] = True
                self.FLAGS['sfc'] = True
            elif self.output_prefix + '_stats.net' in filename and all(char.isdigit() for char in filename[-6:]):
                if self.keep_open :
                    self.output_filenames['stats'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['stats'][self.output_data_dir + filename] = True
                self.FLAGS['stats'] = True
            elif self.output_prefix + '.soilvar.' in filename and all(char.isdigit() for char in filename[-6:]):
                if self.keep_open :
                    self.output_filenames['soil'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['soil'][self.output_data_dir + filename] = True
                self.FLAGS['soil'] = True
            elif self.output_prefix + '.net' in filename and all(char.isdigit() for char in filename[-6:]):
                if self.keep_open :
                    self.output_filenames['hist'][self.output_data_dir + filename] = Dataset(self.output_data_dir + filename, "r")
                else :
                    self.output_filenames['hist'][self.output_data_dir + filename] = True
                self.FLAGS['hist'] = True
        for filename in os.listdir(self.postprocdir):
            if filename.startswith(self.software + self.i_str + "_post_" ):
                if self.keep_open :
                    self.output_filenames['post'][self.postprocdir + filename] = Dataset(self.postprocdir + filename, "r")
                else :
                    self.output_filenames['post'][self.postprocdir + filename] = True
                self.FLAGS['post'] = True
            
        # sort 
        self.output_filenames['stats'] = collections.OrderedDict(sorted(self.output_filenames['stats'].items()))
        self.output_filenames['soil'] = collections.OrderedDict(sorted(self.output_filenames['soil'].items()))
        self.output_filenames['hist'] = collections.OrderedDict(sorted(self.output_filenames['hist'].items()))
        self.output_filenames['post'] = collections.OrderedDict(sorted(self.output_filenames['post'].items()))
              
  
    def init_saved_variables(self):
        #Getting terrain height, soil type and landmask
        self.VARIABLES['HT'].value = HT = self.get_data('HTERAIN')
        SOIL_TYPE = self.get_data("SOIL_TYPE")
        self.VARIABLES['LANDMASK'].value = np.logical_and( SOIL_TYPE != 12, SOIL_TYPE != 13 ) #if soil type isn't "water" or "ice", it is considered as land 
        
        #reading file
        filename = next(iter(self.output_filenames['base']))
        if self.keep_open :
            ncfile = self.output_filenames['base'][filename]
        else :
            ncfile = Dataset(filename, "r")
        #Getting simulation initial time
        self.VARIABLES['INITIAL_TIME'].value = manage_time.to_datetime(ncfile.INITIAL_TIME,fmt="%Y-%m-%d_%H:%M:%S")

        #Getting projection and central coordinates
        self.VARIABLES['MAPPROJ'].value = MAPPROJ = ncfile.MAPPROJ
        self.VARIABLES['SCLFCT'].value = SCLFCT = ncfile.SCLFCT
        self.VARIABLES['CTRLAT'].value = CTRLAT = ncfile.CTRLAT
        self.VARIABLES['CTRLON'].value = CTRLON = ncfile.CTRLON

        #Getting dimensions
        self.VARIABLES['NX'].value = NX = ncfile.dimensions['x'].size
        self.VARIABLES['NY'].value = NY = ncfile.dimensions['y'].size
        self.VARIABLES['NZ'].value = ncfile.dimensions['z'].size
        self.VARIABLES['NZ_SOIL'].value = NZ_SOIL = ncfile.dimensions['zsoil'].size - 1
        self.VARIABLES['NX_XSTAG'].value = NX_XSTAG = ncfile.dimensions['x_stag'].size
        self.VARIABLES['NY_YSTAG'].value = NY_YSTAG = ncfile.dimensions['y_stag'].size
        self.VARIABLES['NZ_ZSTAG'].value = NZ_ZSTAG = ncfile.dimensions['z_stag'].size
        self.VARIABLES['NZ_SOIL_ZSTAG'].value = NZ_SOIL + 1


        #Getting simulation Projection
        TRUELAT1 = ncfile.TRUELAT1
        TRUELAT2 = ncfile.TRUELAT2
        TRUELON = ncfile.TRUELON
        FALSE_EASTING = ncfile.false_easting
        FALSE_NORTHING = ncfile.false_northing
        if abs(MAPPROJ) == 2:
            CRStemp = ccrs.LambertConformal(central_longitude = TRUELON,
                                            central_latitude = TRUELAT1,
                                            false_easting = FALSE_EASTING,
                                            false_northing = FALSE_NORTHING,
                                            standard_parallels = (TRUELAT1, TRUELAT2) )
        else :
            print('ERROR: unknown projection, CRS not done!')
            raise

        #Getting X, Y in simulation Projection
        vec_X_XSTAG = ncfile['x_stag'][:]
        vec_Y_YSTAG = ncfile['y_stag'][:]
        vec_X = manage_list.unstag(vec_X_XSTAG, 0)
        vec_Y = manage_list.unstag(vec_Y_YSTAG, 0)
        X, Y = np.meshgrid(vec_X, vec_Y)
        X_XSTAG, Y_XSTAG = np.meshgrid(vec_X_XSTAG, vec_Y)
        X_YSTAG, Y_YSTAG = np.meshgrid(vec_X, vec_Y_YSTAG)

        #Getting Latitude and Longitude
        self.VARIABLES['LON'].value, self.VARIABLES['LAT'].value = LON, LAT = manage_projection.xy_to_ll(X, Y, CRStemp)
        self.VARIABLES['LON_XSTAG'].value, self.VARIABLES['LAT_XSTAG'].value = LON_XSTAG, LAT_XSTAG = manage_projection.xy_to_ll(X_XSTAG, Y_XSTAG, CRStemp)
        self.VARIABLES['LON_YSTAG'].value, self.VARIABLES['LAT_YSTAG'].value = LON_YSTAG, LAT_YSTAG = manage_projection.xy_to_ll(X_YSTAG, Y_YSTAG, CRStemp)

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
        self.VARIABLES['ZP_ZSTAG'].value = ZP_ZSTAG = ncfile['ZP'][:]
        self.VARIABLES['ZP'].value = manage_list.unstag(ZP_ZSTAG, 0)  
        self.VARIABLES['Z_ZSTAG'].value = Z_ZSTAG = ZP_ZSTAG - np.tensordot(np.ones(NZ_ZSTAG), HT, 0)
        self.VARIABLES['Z'].value = manage_list.unstag(Z_ZSTAG, 0) 
        self.VARIABLES['ZP_SOIL'].value = ZP_SOIL = ncfile['ZPSOIL'][:][1:] #remove surface layer
        self.VARIABLES['Z_SOIL'].value = Z_SOIL = ZP_SOIL - np.tensordot(np.ones(NZ_SOIL), HT, 0)

        a = np.min(np.abs(ZP_ZSTAG[1,:,:] - ZP_ZSTAG[0,:,:]))
        b = np.max(np.abs(ZP_ZSTAG[1,:,:] - ZP_ZSTAG[0,:,:]))
        self.VARIABLES['DZMIN'] = VariableSave("", "", "", "", 0, np.round((a+b)/2)) #taille moyenne de la premiere maille
        a = np.min(np.abs(Z_ZSTAG[1:] - Z_ZSTAG[:-1]))
        b = np.max(np.abs(Z_ZSTAG[1:] - Z_ZSTAG[:-1]))
        self.VARIABLES['DZ'] = VariableSave("", "", "", "", 0, np.round((a+b)/2)) #taille moyenne des mailles
        a = np.min(np.abs(ZP_ZSTAG[1:,:,:] - ZP_ZSTAG[:-1,:,:]))
        b = np.max(np.abs(ZP_ZSTAG[1:,:,:] - ZP_ZSTAG[:-1,:,:]))
        self.VARIABLES['DZMAX'] = VariableSave("", "", "", "", 0, np.round((a+b)/2)) #taille moyenne de la derniere maille
        self.VARIABLES['DZ_SOIL'].value = - np.diff(Z_SOIL[:, 0, 0])
        self.init_stats_variables()
        if not self.keep_open :
            ncfile.close()
        
    def init_stats_variables(self):
        if self.FLAGS["stats"] :
            filename = next(iter(self.output_filenames['stats']))
            ncfile = self.output_filenames['stats'][filename]
            self.VARIABLES['NMK'].value = NMK = ncfile.dimensions['mk'].size 
            all_varnames = self.VARIABLES.keys()
            all_MK_varnames = []
            for k in all_varnames:
                if k.startswith("MK") :
                    all_MK_varnames.append(k)
            for k in all_MK_varnames :
                varname_quantity = k[2:]
                if varname_quantity.startswith("CB") :
                    continue
                var_quantity = self.VARIABLES[varname_quantity]
                latex_units = var_quantity.latex_units
                units = var_quantity.units
                dim = var_quantity.dim
                cmap = self.VARIABLES[k].cmap
                for i_K in range(NMK) :
                    varname = "M"+str(i_K)+ varname_quantity
                    legend_short = 'M'+str(i_K+1)+'('+var_quantity.legend_short+')'
                    legend_long = ["1st", "2nd", "3rd", "4th", "5th"][i_K] + " order moment of "+ var_quantity.legend_short
                    self.VARIABLES[varname] = Variable(legend_short, legend_long, latex_units, units, dim, cmap)
            
    def set_aerradii(self): # A vérifier
        if self.aerradii is None:
            aerradii_dom = []
            aerrho_dom = []
            aertyp_dom = []
             
            ncfile = list(self.output_filenames['base'].values())[0]
            try:
                aerradii_dom = ncfile['aerradii'][:]
                aerrho_dom = ncfile['aerrho'][:]
                aertyp_dom = ncfile['aertyp'][:]
                aerradii = 1
            except IndexError:
                self.aerradii = None
            if self.aerradii is None:
                self.aerradii = None
                self.aerrho = None
                self.aertyp = None
            else:
                self.FLAGS['conc'] = True
                self.aerradii = aerradii_dom
                self.aerrho = aerrho_dom
                self.aertyp = aertyp_dom
                if self.aerradii.size != self.aerrho.size:
                    print('ERROR: aerosol radii not consistent with size')
                    raise
        else:
            self.aerradii = np.array(aerradii) #array ?

        if self.aerradii is not None:
            self.FLAGS['conc'] = True
            self.VARNAMES['timevar3D'].append('CB')
            for i in range(1,self.aerradii.size+1):
                self.VARNAMES['timevar3D'].append('CB_{0}'.format(str(i).zfill(3)))
        else:
            self.FLAGS['conc'] = False
    
    def calculate(self, varname, **kwargs):
        if varname in ["U", "U_BASE", "U_PERT"]:
            kwargs_temp = copy.copy(kwargs)
            kwargs_temp["i_unstag"] = 2
            return self.get_data(varname+"_XSTAG", **kwargs_temp)
        elif varname == "U_PERT_XSTAG":
            return self.get_data("U_XSTAG", **kwargs) - self.get_data("U_BASE_XSTAG", **kwargs)
        elif varname in ["V", "V_BASE", "V_PERT"]:
            kwargs_temp = copy.copy(kwargs)
            kwargs_temp["i_unstag"] = 1
            return self.get_data(varname+"_YSTAG", **kwargs_temp)
        elif varname == "V_PERT_YSTAG":
            return self.get_data("V_YSTAG", **kwargs) - self.get_data("V_BASE_YSTAG", **kwargs)
        elif varname in ["W", "W_BASE", "W_PERT"]:
            kwargs_temp = copy.copy(kwargs)
            kwargs_temp["i_unstag"] = 0
            return self.get_data(varname+"_ZSTAG", **kwargs_temp)
        elif varname == "W_PERT_ZSTAG":
            return self.get_data("W_ZSTAG", **kwargs) - self.get_data("W_BASE_ZSTAG", **kwargs)
        elif varname in ["PT_PERT", "P_PERT", "QV_PERT"]:
            varname_inst = varname[:-5]
            varname_base = varname_inst + "_BASE"
            var_TOTAL = self.get_data(varname[:-5], **kwargs)
            var_BASE = self.get_data(varname[:-5]+"_BASE", **kwargs)
            return var_TOTAL - var_BASE 
        elif varname in ["T", "T_BASE", "T_PERT"]: #definition of T_PERT, T_BASE is ambiguous, use with care
            PT_var = self.get_data("P"+varname, **kwargs)
            P = self.get_data("P", **kwargs)
            return constants.PT_to_T(PT_var, P)
        elif varname == "U_STAR":
            print("error : The calculation of U_STAR hasn't been added to class_domARPS/calculate")
            #raise
        elif varname in ["T_SOIL", "Q_SOIL"]:
            var_raw = self.get_data(varname[0]+"SOIL", **kwargs)#"TSOIL" or "QSOIL"
            return var_raw[0][1:] #return only mean value and not for each soil type, and remove surface layer
        elif varname in ["T_SKIN", "Q_SKIN"]:
            var_raw = self.get_data(varname[0]+"SOIL", **kwargs)#"TSOIL" or "QSOIL"
            return var_raw[0][0] #return only mean value and not for each soil type, only surface layer
        elif varname == "SOIL_TYPE" :
            SOILTYP = self.get_data("SOILTYP", **kwargs)
            return SOILTYP[0] #return only main soil type
        elif varname == "POROSITY": #could be better by using all the soil types per cell and not only the main type
            tab_porosity = [np.nan, 0.417, 0.421, 0.434, 0.486, 0.439, 0.404, 0.464, 0.465, 0.406, 0.423, 0.468, 0,     0] 
            SOIL_TYPE = self.get_data("SOIL_TYPE", **kwargs)
            POROSITY = np.zeros(SOIL_TYPE.shape)
            for i in range(1, 14) :
                POROSITY[np.where(SOIL_TYPE == i)] = tab_porosity[i]
            return POROSITY
        elif varname == "WSAT":
            tab_wsat = [np.nan, 0.421, 0.421, 0.434, 0.434, 0.439, 0.404, 0.464, 0.465, 0.406, 0.406, 0.468, 1e-20, 1]
            SOIL_TYPE = self.get_data("SOIL_TYPE", **kwargs)
            WSAT = np.zeros(SOIL_TYPE.shape)
            for i in range(1, 14) :
                WSAT[np.where(SOIL_TYPE == i)] = tab_wsat[i]
            return WSAT
        elif varname == "QUARTZ":
            tab_quartz =   [np.nan, 0.92,  0.82,  0.6,   0.25,  0.4,   0.6,   0.1,   0.35,  0.52,  0.1,   0.25,  0,     0] 
            SOIL_TYPE = self.get_data("SOIL_TYPE", **kwargs)[0] #use only main type
            QUARTZ = np.zeros(SOIL_TYPE.shape)
            for i in range(1, 14) :
                QUARTZ[np.where(SOIL_TYPE == i)] = tab_quartz[i]
            return QUARTZ
        elif varname[0] == "M" and len(varname) > 1 and varname[1].isnumeric() : #Moment of order varname[1]
            i_K = int(varname[1])
            assert i_K < self.get_data("NMK")
            kwargs_MK = copy.copy(kwargs)
            crop = (i_K,) + kwargs_MK["crop"] if kwargs_MK["crop"] is not None else (i_K, "ALL", "ALL", "ALL")
            kwargs_MK["crop"] = crop
            return self.get_data("MK"+varname[2:], **kwargs_MK)
        else :
            return Dom.calculate(self, varname, **kwargs)
                
