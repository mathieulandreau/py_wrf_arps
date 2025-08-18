import numpy as np
from multiprocessing import Pool
import scipy
import os
import netCDF4
import datetime
#import copy
#from matplotlib import pyplot as plt
from ..lib import manage_scipy, manage_time, manage_angle, constants
from ..WRF_ARPS import Dom
from ..class_proj import Proj

class SBTKE():
    def __init__(self, sim, dom, itime=("2020-05-18-10", "2020-05-18-18", "4h"), NZ=30, figdir="t08/manuscript/07/"):
        self.sim = sim
        self.dom = self.sim.get_dom(dom)
        self.postprocdir = self.sim.postprocdir+self.dom.name
        self.itime = itime
        self.IT, self.TIME = self.dom.get_data(["IT", "TIME"], itime=itime)
        self.NT = len(self.TIME)
        self.NZ = NZ
        self.figdir = figdir
        self.p = {} #contains average values for each zones with shape (NT, NZ, Nzones)
        self.p2 = {} #same as p but with shape (NT, NZ, NY, NX)
        self.create_zones()
        self.dist2D, self.TIME2D = np.meshgrid(self.dist, self.TIME)
        self.HOUR2D = np.array(manage_time.to_hour(self.TIME2D))
        
    def create_zones(self):
        DX, DY, NX, NY = self.dom.get_data(["DX", "DY", "NX", "NY"])
        boundary_dist = 1e3
        ix0 = int(boundary_dist//DX)
        iy0 = int(boundary_dist//DY)
        crop = ([0, self.NZ], [iy0, NY-iy0], [ix0, NX-ix0])
        self.crop = crop
        if os.path.exists(self.postprocdir+"_SBTKE_zones.npy"):
            print("load zones")
            self.zones = np.load(self.postprocdir+"_SBTKE_zones.npy")
            self.dist  = np.load(self.postprocdir+"_SBTKE_dist.npy")
            self.Nzones = len(np.unique(self.zones)) - 1
        else :
            Y, X, CDI, LANDMASK, BDI = self.dom.get_data(["Y_KM", "X_KM", "CDI_SIGMA", "LANDMASK", "BDY_DIST"], sigma=1, crop=crop)
            zones = (np.round((-CDI)).astype(int) + 7)*7 + np.round((Y-X)/3 + 3.5)
            zones[np.logical_or(Y-X > 9, Y-X < -12)] = -1
            zones[zones < -1] = -1
            zones[-CDI>18] = -1
            zones[BDI<1e3] = -1
            # keep only zones that have more than 1000 cells in it
            temp = np.unique(zones)
            for izone in temp :
                if np.sum(zones==izone) < 1000:
                    zones[zones==izone] = -1
            # give a continuous range of numbers to the zones
            temp = np.unique(zones)
            zones2 = np.copy(zones)
            for i, izone in enumerate(temp) :
                zones2[zones==izone] = i-1
            zones = zones2
            # calculate distance from the 1-km coastline of each zone
            Nzones = len(np.unique(zones)) - 1
            dist = np.zeros((Nzones))*np.nan
            for izone in range(Nzones):
                dist[izone] = np.mean(CDI[zones==izone])
            zones[zones == -1] = np.nan
            self.dist = dist
            self.zones = zones
            self.Nzones = Nzones
            np.save(self.postprocdir+"_SBTKE_zones.npy", self.zones)
            np.save(self.postprocdir+"_SBTKE_dist.npy", self.dist)

#################################################################################################################################
######  GET DATA
################################################################################################################################# 
    def get_data_it(self, varnames, it, itime, typ="mean"):
        p2 = {}
        self.dom.get_data(varnames, itime=itime, crop=self.crop, saved=p2, return_val=False)
        for izone in range(self.Nzones):
            mask = self.zones==izone
            for v in varnames :
                dim = self.get_dim(v)
                if typ == "std" : 
                    if dim == 2:
                        self.p["STD_"+v][it, izone] = np.nanstd(p2[v][0, mask], axis=-1)
                    else :
                        self.p["STD_"+v][it, :, izone] = np.nanstd(p2[v][:, mask], axis=-1)
                elif typ == "mean" :
                    if dim == 2:
                        self.p[v][it, izone] = np.nanmean(p2[v][0, mask], axis=-1)
                    else :
                        self.p[v][it, :, izone] = np.nanmean(p2[v][:, mask], axis=-1)
                else : 
                    raise(Exception(f"unkown typ {typ}"))
        
    def get_data(self, varnames, nprocs=1, typ="mean"):
        for v in varnames :
            dim = self.get_dim(v)
            if not v in self.p:
                if dim == 2:
                    self.p["STD_"*(typ=="std")+v] = np.zeros((self.NT, self.Nzones))
                else:
                    self.p["STD_"*(typ=="std")+v] = np.zeros((self.NT, self.NZ, self.Nzones))
        # get the data and compute the average of each zones
        if nprocs == 1:
            for it, itime in enumerate(self.IT) :
                self.get_data_it(varnames, it, itime, typ)
        #does not work for now
        else:
            inputs = [(varnames, it, itime, typ) for it, itime in enumerate(self.IT)]
            with Pool(processes=nprocs) as pool:
                pool.starmap(self.get_data_it, inputs)
    
    def get_dim(self, varname):
        if varname in ["Zpeak", "TKEpeak", "IZpeak", "SBZC"]:
            return 2
        else :
            return self.dom.get_dim(varname)
    
#################################################################################################################################
######  Analysis
################################################################################################################################# 
    def var_to_map(self, varnames):
        zonestemp = self.zones.astype(int)
        mask = np.isnan(self.zones)
        zonestemp[mask] = 0
        for v in varnames :
            if self.get_dim(v) == 2:
                self.p2[v] = self.p[v][:, zonestemp]
                self.p2[v][:, mask] = np.nan
            else :
                print(v)
                self.p2[v] = self.p[v][:, :, zonestemp]
                self.p2[v][:, :, mask] = np.nan

    def find_peaks(self, prom_min=0.03, Z_min=0, Z_max=400):
        PROM = np.apply_along_axis(manage_scipy.peak_prominences, 1, self.p["TKE"], np.arange(self.NZ))
        PROM[PROM<prom_min]=0
        PROM[self.p["Z"]>Z_max] = 0
        PROM[self.p["Z"]<Z_min] = 0
        TKEtemp = np.copy(self.p["TKE"])
        Ztemp = np.copy(self.p["Z"])
        TKEtemp[PROM<prom_min/2] = 0
        Ztemp[PROM<prom_min/2] = np.nan
        IZpeak = np.expand_dims(np.nanargmax(TKEtemp, axis=1), axis=1)
        TKEpeak = np.take_along_axis(TKEtemp, IZpeak, axis=1)
        Zpeak = np.take_along_axis(Ztemp, IZpeak, axis=1)
        self.p["Zpeak"] = np.squeeze(Zpeak)
        self.p["TKEpeak"] = np.squeeze(TKEpeak)
        self.p["IZpeak"] = np.squeeze(IZpeak)
        
    def find_wave_angle(self, save=True):
        kwt = {
            "crop" : self.crop,
            "itime" : self.itime, 
            "saved" : {},
        }
        A = self.dom.get_data("W_INST", **kwt)
        grad_y, grad_x = np.gradient(A, 1, 1, axis=(-2, -1))
        grad = np.sqrt(grad_x**2 + grad_y**2)
        angles = manage_angle.UV2WD_deg(grad_x, grad_y)
        angles = angles%180
        mean_angle = np.zeros((self.NT, self.NZ, self.Nzones))
        std_angle = np.zeros((self.NT, self.NZ, self.Nzones))
        for izone in range(self.Nzones) :
            print(izone, end=",")
            grad_temp = grad[:, :, self.zones==izone]
            angles_temp = angles[:, :, self.zones==izone]
            mean_angle[:,:,izone] = manage_angle.circmean(angles_temp, weights=grad_temp, high=180, axis=-1, nan_policy='omit')
            std_angle[:,:,izone] = manage_angle.circstd(angles_temp, weights=grad_temp, high=180, axis=-1, nan_policy='omit')
        self.p["WAVE_ANGLE"] = mean_angle
        self.p["STD_WAVE_ANGLE"] = std_angle
        if save :
            for v in ["WAVE_ANGLE", "STD_WAVE_ANGLE"]:
                self.write_postproc(v)
        self.get_data(["W_INST"], typ="std")
        if save :
            self.write_postproc("STD_W_INST")

    def calculate_shear_RI(self, save=True):
        for v in ["Z", "U", "V", "PTV_C"] :
            if not v in p :
                self.read_postproc()
            if not v in p :
                self.get_data([v])
                if save :
                    self.write_postproc(v)
        gradZ = np.gradient(self.p["Z"], axis=1)
        num = constants.BETA * np.gradient(self.p["PTV_C"], axis=1)/gradZ
        den = (np.gradient(self.p["U"], axis=1)/gradZ)**2 + (np.gradient(self.p["V"], axis=1)/gradZ)**2
        self.p["RI"] = num/den
        self.p["SHEAR"] = np.sqrt(den)
        self.p["SHEAR_DIR"] = manage_angle.UV2WD_deg(np.gradient(self.p["U"], axis=1)/gradZ, np.gradient(self.p["V"], axis=1)/gradZ)
        self.p["NBV2"] = num
        if save :
            self.write_postproc("RI")
            self.write_postproc("SHEAR")
            self.write_postproc("SHEAR_DIR")
            self.write_postproc("NBV2")
    
    def calculate_KH_index(self, save=True):
        if not "WAVE_ANGLE" in self.p :
            self.find_wave_angle()
        if not "SHEAR_DIR" in self.p :
            self.calculate_shear_RI()
        diff = np.abs((self.p["WAVE_ANGLE"]-self.p["SHEAR_DIR"]+90)%180 - 90)
        self.p["KH1"] = 1.0*np.logical_and(diff < 20, self.p["STD_W_INST"] > 0.1)
        if save :
            self.write_postproc("KH1")

#################################################################################################################################
######  PLOT FIGURES
################################################################################################################################# 
    def plot_zones(self):
        zonestemp = self.zones.astype(int)
        zonestemp[np.isnan(self.zones)] = 0
        disttemp = self.dist[zonestemp]
        disttemp[np.isnan(self.zones)] = np.nan
        params = [{ 
            "typ" : "2DH", "dom" : self.dom, "cmap" : "Purples_r", "Z" : disttemp, "clim" : [-16, 4], "discrete" : 5, "plot_cbar" : False, 
            "kwargs_get_data" : {"crop" : (0, self.crop[1], self.crop[2])}, "title" : "",
            "dpi" : 120, "savepath" : f"{self.figdir}TKEpeak_Zones{self.Nzones}", "kwargs_LANDMASK" : {"kwargs_plt" : {"colors" : [[.5, .5, .5]], "linewidths" : [2]},},
        },{ "typ" : "CONTOUR", "Z" : zonestemp, "kwargs_plt" : {"levels" : np.arange(self.Nzones)}, "clabel" : False,
        },{ "typ" : "MASK", "X" : "X_KM", "Y" : "Y_KM", "Z" : "LANDMASK_SIGMA", "kwargs_get_data" : {"sigma" : 1}, "kwargs_plt" : {"linewidths" : [2], "colors" : [[1, 0, 0]]}
        }]
        self.sim.plot_fig(params)
    
    def plot_peaks(self):
        zones = np.arange(self.Nzones)
        dec = ((zones*13)%12)/12
        dec2 = ((zones%13)//12)/12
        mask2 = self.p["SBZC"] > 1
        mask = np.logical_and(self.dist2D<-2, mask2)
        params = [{
            "X" : (self.dist2D+1.5*(.5-dec))[mask2], "Y" : (self.HOUR2D+.1*(.5-dec))[mask2]*100, "Z" : self.p["TKEpeak"][mask2]>0.0001, 
            "xlim" : [-15.5, 7], "ylim" : [1190, 1810], "xticks" : np.arange(-15, 7, 3), "DX_subplots" : 10,
            "cmap" : "Reds", "kwargs_plt" : {"s" : 40, "edgecolor" : "k", "linewidth" : .5}, "plot_cbar" : False, "clim" : [0, 1.8], 
            "xlabel" : "$X_c$ (km)", "ylabel" : "Hour (UTC)",
            "dpi" : 120,  "savepath" : f"{self.figdir}TKE_peak_presence_dist_TIME",
        },{ "typ" : "AXVLINE",
        },{ "typ" : "AXVLINE", "X" : -2,

        },{ "X" : self.p["SBZC"][mask], "Y" : self.p["Zpeak"][mask]+10*(.5-np.random.random(np.sum(mask))), "Z" : self.dist2D[mask], "style" : ".", 
            "kwargs_plt" : {"s" : self.p["TKEpeak"][mask]*100, "edgecolor" : "k", "linewidth" : .5}, 
            "cmap" : "Purples_r", "clim" : [-16, 4], "discrete" : 5, "xlim" : [0, 300], "ylim" : [0, 300], "DX_subplots" : 10,
            "xlabel" : "$Z_{SB}$ (m)", "ylabel" : "Height of the $k_{tot}$ peak (m)", "clabel" : "$X_c$ (km)",
            "same_fig" : False, "dpi" : 120, "savepath" : f"{self.figdir}TKE_peak_Zpeak_Zc_dist"
        },{ "X" : [0, 300], "Y" : [0, 300], "style":"k", "same_ax" : True,
        },{ "X" : np.array([-1, -1]), "Y" : np.array([-100, -100]), "Z" : np.array([0, 0]), "cmap" : "Greys_r", "clim" : [0, 1],
            "kwargs_plt" : {"s" : [50]}, "label" : "0.5 m$^2$.s$^{-2}$", "same_ax" : True, "plot_cbar" : False,
        },{ "X" : np.array([-1, -1]), "Y" : np.array([-100, -100]), "Z" : np.array([0, 0]), "cmap" : "Greys_r", "clim" : [0, 1],
            "kwargs_plt" : {"s" : [10]}, "label" : "0.1 m$^2$.s$^{-2}$", "same_ax" : True, "plot_cbar" : False, "legend_loc" : "lower left",

        },{ "same" : -4, "Z" : self.HOUR2D[mask]*100,
            "cmap" : "Oranges", "clim" : [1200, 1800], "discrete" : 6, "clabel" : "Hour (UTC)",
            "savepath" : f"{self.figdir}TKE_peak_Zpeak_Zc_time",
        },{ "same" : -4,
        }]
        return self.sim.plot_fig(params)

#################################################################################################################################
######  READ AND WRITE POSTPROC
#################################################################################################################################  
    def init_postproc(self):
        for it in range(self.NT):
            filename = self.it_to_filename(it)
            if not os.path.exists(filename):
                with netCDF4.Dataset(filename, mode="w", format='NETCDF4_CLASSIC') as ncfile : 
                    ncfile.Title        = "py_wrf_arps SBTKE postproc"
                    ncfile.Institution  = "LHEEA-DAUC"
                    ncfile.FMTVER       = "NetCDF 4.0 Classic"
                    ncfile.Source       = "py_wrf_arps/post/class_SBTKE.py"
                    ncfile.References   = "See class_SBTKE.py"
                    ncfile.Comment      = "Postproc data saved in file"
                    now = datetime.datetime.now()
                    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                    ncfile.History      = "Creation date: {date}".format(date=dt_string)
                    ncfile.createDimension('Nzones', self.Nzones)
                    ncfile.createDimension('NZ', self.NZ)
                    ncfile.DATE = self.dom.date2str(self.dom.date_list[self.IT[it]], "WRF")


    def write_postproc(self, varname, dim=None, long_name=None, standard_name=None, units="", latex_units=None, datatyp=np.float32):
        # Prepare
        self.init_postproc()
        if dim is None :
            dim = self.p[varname].ndim - 1
        if dim == 2:
            dims = ("NZ", "Nzones")
        elif dim == 1:
            dims = ("Nzones",)
        else :
            raise(Exception(f"Unknown dim : {dim}"))
        if standard_name is None :
            if varname in self.dom.VARIABLES:
                long_name = self.dom.VARIABLES[varname].legend_long
                standard_name = self.dom.VARIABLES[varname].legend_short
                units = self.dom.VARIABLES[varname].units
                latex_units = self.dom.VARIABLES[varname].latex_units
            else:
                long_name = varname
                standard_name = varname
        if latex_units is None:
            latex_units = units
        print(f"dim : {dim}, long_name : {long_name}, standard_name : {standard_name}, units : {units}, latex_units : {latex_units}, datatyp : {datatyp}")
        # Write
        mode = "a"
        for it in range(self.NT):
            filename = self.it_to_filename(it)
            ncfile = netCDF4.Dataset(filename, mode=mode, format='NETCDF4_CLASSIC')
            if mode == "a":
                if varname in ncfile.variables :
                    ncfile.close()
                    ncfile = netCDF4.Dataset(filename, mode="r+", format='NETCDF4_CLASSIC')
                    ncvar = ncfile[varname]
                    mode = "r+"
                    print(f"mode = r+ : {varname} already exist in the file")
                else :
                    ncvar = ncfile.createVariable(varname, datatyp, dims)
            if mode == "r+":
                if not varname in ncfile.variables :
                    ncfile.close()
                    ncfile = netCDF4.Dataset(filename, mode="a", format='NETCDF4_CLASSIC')
                    ncvar = ncfile.createVariable(varname, datatyp, dims)
                    mode = "a"
                    print(f"mode = a : {varname} do not exist in the file")
                else :
                    ncvar = ncfile[varname]
            ncvar.long_name = varname
            ncvar.standard_name = varname
            ncvar.units = units
            ncvar.latex_units = latex_units
            ncvar[:] = self.p[varname][it]
            ncfile.close()
            
    def read_postproc(self, varnames=None):
        it = 0
        filename = self.it_to_filename(it)
        if os.path.exists(filename):
            ncfile = netCDF4.Dataset(filename, mode="r")
            if varnames is None :
                varnames = [v for v in ncfile.variables]
            elif type(varnames) is str :
                varnames = [varnames]
            for v in varnames :
                dim = self.get_dim(v)
                if dim == 2:
                    self.p[v] = np.zeros((self.NT, self.Nzones))
                else:
                    self.p[v] = np.zeros((self.NT, self.NZ, self.Nzones))
            ncfile.close()
            for it in range(self.NT):
                filename = self.it_to_filename(it)
                ncfile = netCDF4.Dataset(filename, mode="r")
                for v in varnames :
                    self.p[v][it] = ncfile[v][:]
                ncfile.close()
        else :
            print("No postproc file found")
    
    def it_to_filename(self, it) :
        itime = self.IT[it]
        date_str = self.dom.date2str(self.dom.date_list[itime], "WRF")
        return f"{self.postprocdir}_SBTKE{date_str}.nc"
        
    def filename_to_it(self, filename) :
        date = manage_time.to_datetime(filename[-23:-4])
        itime = self.dom.date_list.index(date)
        return self.IT.index(itime)

#################################################################################################################################
######  COMPLETE PROCEDURE
#################################################################################################################################    
    def compute_peak_and_save_data_for_the_first_time(self):
        # fig = self.plot_zones()
        varnames = ["Z", "TKE_RES", "TKE", "AD225_U"]
        self.get_data(varnames)
        self.p["SBZC"] = np.squeeze(self.dom.get_Z_SB_LB(self.p["AD225_U"], self.p["Z"], zaxis=1))
        self.find_peaks()
        for v in varnames + ["SBZC", "Zpeak", "IZpeak", "TKEpeak"] :
            self.write_postproc(v)
        # fig = self.plot_peaks()
            