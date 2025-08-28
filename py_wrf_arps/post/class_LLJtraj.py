import numpy as np
import scipy
import datetime
import copy
import os
import netCDF4
from matplotlib import pyplot as plt
from ..lib import manage_images, manage_time, manage_traj, manage_angle, manage_plot
from ..WRF_ARPS import Dom
from ..class_proj import Proj

class LLJtraj():
    
    def __init__(self, sim, dom, itime=("2020-05-17-15", "2020-05-18-08"), itstr_list=["2020-05-18-00", "2020-05-18-01", "2020-05-18-02"], cmap="turbo"):
        self.sim = sim
        self.dom = self.sim.get_dom(dom)
        self.cmap = cmap
        self.date0_list = manage_time.to_datetime(itstr_list)
        self.NT0 = len(self.date0_list)
        self.NZ = 21
        # Raw variables
        self.p = {}
        self.kw = {
            "itime" : itime,
            "crop" : ([0, self.NZ], "ALL", "ALL"),
            "saved" : self.p,
            "quick_deriv" : True,
        }
        # Horizontally filtered variables
        self.ps = {}
        self.kws = copy.deepcopy(self.kw)
        self.kws["DX_smooth"] = int(10*self.dom.get_data("DX")/1000)
        self.kws["sigma"] = int(10*self.dom.get_data("DX")/1000)
        self.kws["saved"] = self.ps
        self.TIMEin = self.dom.get_data("TIME", **self.kw)
        self.NT = len(self.TIMEin)
        self.X, self.Y, self.DX, self.DY, self.IY, self.IX = self.dom.get_data(["X_KM", "Y_KM", "DX", "DY", "IY", "IX"])
        self.pt = {}
        self.pp = {}
        self.pm = {}
        self.NY, self.NX = self.Y.shape
        self.get_initial_position()
        
    def get_initial_position(self):
        self.lm = self.dom.get_data("LANDMASK_SIGMA", **self.kws)[0]
        self.lm = 1*(self.lm>0.5)
        COASTCELL = manage_images.get_COASTCELL(self.lm)
        COASTCELL[self.Y > 60] = 0
        IX2D, IY2D = np.meshgrid(self.IX, self.IY)
        self.ix_init = IX2D[COASTCELL==1]
        self.iy_init = IY2D[COASTCELL==1]
        self.Ntraj = len(self.ix_init)
        self.iz_init = np.array([5]*self.Ntraj)
        self.it_init_in = np.zeros((self.NT0, self.Ntraj), dtype=int)
        self.pt["DELTA"] = np.zeros((self.NT0, self.Ntraj, self.NT))
        for it0 in range(self.NT0):
            self.it_init_in[it0] = np.array([manage_time.get_time_slice(self.date0_list[it0], self.TIMEin)]*self.Ntraj)
            TIME, _ = np.meshgrid(self.TIMEin, np.arange(self.Ntraj))
            self.pt["DELTA"][it0] = manage_time.timedelta_to_seconds(TIME - self.date0_list[it0])/3600

#################################################################################################################################
######  GET DATA
################################################################################################################################# 

    def get_traj_data(self):
        # Trajectories are computed with horizontally filtered U, V, W, ZP, HT, DZ
        varnames = ["U", "V", "W", "Z", "DZ", "LANDMASK_SIGMA", "ZP", "HT"]
        _ = self.dom.get_data(varnames, return_val=False, print_level=2, **self.kws)
        # X, Y, ... are not filtered
        varnames2 = ["LAT", "LON", "LANDMASK"]
        _ = self.dom.get_data(varnames2, return_val=False, **self.kw)
    
    def get_sdata(self, varnames):
        self.dom.get_data(varnames, return_val=False, print_level=2, **self.kws)
        
    def get_data(self, varnames):
        self.dom.get_data(varnames, return_val=False, print_level=2, **self.kw)
        
#################################################################################################################################
######  TRAJECTORY COMPUTATION
################################################################################################################################# 

    def init_traj_variables(self):
        for v in ["X", "Y", "ZP", "IX", "IY", "IZ"] :
            self.pt[v] = np.zeros((self.NT0, self.Ntraj, self.NT))
        
    def calculate_traj(self):
        for it0 in range(self.NT0) :
            self.pt["X"][it0], self.pt["Y"][it0], self.pt["ZP"][it0], self.pt["IX"][it0], self.pt["IY"][it0], self.pt["IZ"][it0] = \
                manage_traj.calculate_double_traj(self.X, self.Y, self.ps["ZP"], self.ps["U"], self.ps["V"], self.ps["W"], self.ps["HT"][0], self.DX, self.DY, self.ps["DZ"], 
                                                  self.TIMEin, self.ix_init, self.iy_init, self.iz_init, self.it_init_in[it0], dt_small="1m")
    
    def raw_to_profile(self, varnames):
        if type(varnames) in [str] : varnames = [varnames]
        for v in varnames:
            if not (v in self.p or v in self.ps) :
                print(f"Error : the varname {v} has not been loaded yet, cannot compute LLJtraj.raw_to_profile")
            elif v not in self.pp : 
                self.pp[v] = np.zeros((self.NT0, self.Ntraj, self.NT, self.NZ))
                for it0 in range(self.NT0) : 
                    if v in self.ps :
                        self.pp[v][it0] = manage_traj.traj_to_profile(self.pt["IX"][it0], self.pt["IY"][it0], self.ps[v])
                    elif v in self.p :
                        self.pp[v][it0] = manage_traj.traj_to_profile(self.pt["IX"][it0], self.pt["IY"][it0], self.p[v])
             
    def raw_to_traj(self, varnames): 
        if type(varnames) in [str] : varnames = [varnames] 
        for v in varnames:
            if not (v in self.p or v in self.ps) :
                print(f"Error : the varname {v} has not been loaded yet, cannot compute LLJtraj.raw_to_profile")
            elif v not in self.pt :  
                self.pt[v] = np.zeros((self.NT0, self.Ntraj, self.NT))
                for it0 in range(self.NT0) : 
                    if v in self.ps :
                        dim = np.squeeze(self.ps[v]).ndim
                        if dim == 4 :
                            self.pt[v][it0] = manage_traj.traj_to_var(self.pt["IX"][it0], self.pt["IY"][it0], self.pt["IZ"][it0], self.ps[v])
                        elif dim == 3 :
                            self.pt[v][it0] = manage_traj.traj_to_var2D(self.pt["IX"][it0], self.pt["IY"][it0], np.squeeze(self.ps[v]))
                        else :
                            raise(Exception(f"Wrong dim ({dim}) of self.ps[{v}]"))
                    elif v in self.p :
                        dim = np.squeeze(self.p[v]).ndim
                        if dim == 4 :
                            self.pt[v][it0] = manage_traj.traj_to_var(self.pt["IX"][it0], self.pt["IY"][it0], self.pt["IZ"][it0], self.p[v])
                        elif dim == 3 :
                            self.pt[v][it0] = manage_traj.traj_to_var2D(self.pt["IX"][it0], self.pt["IY"][it0], np.squeeze(self.p[v]))
                        else :
                            raise(Exception(f"Wrong dim ({dim}) of self.p[{v}]"))
                            
    def traj_to_map(self, varnames):
        for v in ["LAT", "LON"]:
            if not v in self.pt: self.read_postproc(["t"+v])
        if type(varnames) is str : varnames = [varnames]
        for v in varnames :
            if v not in self.pm :  
                self.pm[v] = np.zeros((self.NT0, self.NY, self.NX))
                for it0 in range(self.NT0) :
                    print(v, it0)
                    self.pm[v][it0] = self.dom.interpolate_to_self_grid(self.pt["LAT"][it0], self.pt["LON"][it0], self.pt[v][it0], max_dist_km = 5*self.dom.get_data("DX")/1000)

#################################################################################################################################
######  PLOT FIGURES
################################################################################################################################# 

    def plot_traj(self, selected_traj=[200, 400, 600], DIR=None):
        if DIR == None :
            DIR = f"t{self.sim.tab_test[0]}/LLJ/traj/"
        Nselect = len(selected_traj)
        for it0 in range(self.NT0) :
            itstr = self.it0_to_datestr(it0)
            TIME_init = manage_time.to_datetime(itstr)
            ticklabels = manage_time.date_to_str(manage_time.to_datetime(manage_time.to_datetime64(TIME_init) + np.arange(-9, 9.1, 3).astype('timedelta64[h]')), "UTC")
            it_init_in = self.it_init_in[it0, 0]
            selected_i_it = range(it_init_in%6, self.NT, 6)
            params = [{ "typ" : "2DH", "dom" : self.dom, "Z" : "LANDMASK", "title" : "", "plot_cbar" : False,
            },{ "typ" : "MASK", "Z" : self.lm, "kwargs_plt" : {"colors" : [[0, 1, 0]], "linewidths" : [3]},
            },{ "typ" : "DATE", "kwargs_get_data" : {"itime" : itstr}, "kwargs_plt" : {"color" : [0.7, 0.7, 0.7],},
            }]
            for i_traj in selected_traj:
                params.append({
                    "same_ax" : True, "X" : self.pt["X"][it0,i_traj], "Y" : self.pt["Y"][it0,i_traj], "Z" : self.pt["DELTA"][it0,i_traj], "cmap" : self.cmap, "discrete" : 6, 
                    "grid" : False, "clim" : [-9, 9], "kwargs_plt" : {"s" : 10, "edgecolor" : "w", "linewidth" : 0.5},
                    "plot_cbar" : i_traj == selected_traj[0], "clabel" : "Hour (UTC)", "ticks" : np.arange(-9, 9.1, 3), "ticklabels" : ticklabels,
                    "dpi" : 120, "savepath" : f"{DIR}04_map_{Nselect}traj_tc{itstr}",
                })
                params.append({
                    "same_ax" : True, "X" : self.pt["X"][it0,i_traj,selected_i_it], "Y" : self.pt["Y"][it0,i_traj,selected_i_it], "Z" : self.pt["DELTA"][it0,i_traj,selected_i_it], 
                    "cmap" : self.cmap, "discrete" : 6, "grid" : False, "clim" : [-9, 9], "kwargs_plt" : {"edgecolor" : "w", "linewidth" : .5}, "plot_cbar" : False,
                })
            fig = self.sim.plot_fig(params)
    
    def plot_profile(self, varname, selected_traj=[200, 400, 600], selected_t0=None, xlim=None, xlabel=None, DIR=None):
        if DIR == None :
            DIR = f"t{self.sim.tab_test[0]}/LLJ/traj/"
        if selected_t0 is None : selected_t0 = range(self.NT0)
        if type(selected_t0) in [int, np.int64] : selected_t0 = [selected_t0]
        for it0 in selected_t0 :
            itstr = self.it0_to_datestr(it0)
            it_init_in = self.it_init_in[it0, 0]
            selected_i_it = {
                "before" : range(it_init_in%6, it_init_in+1, 6),
                "after" : range(it_init_in, self.NT, 6),
            }
            for str0 in ["before", "after"]:
                vmin, vmax, extend, cmap, ticks = manage_plot.get_cmap_extend([-9.08, 8.92], np.arange(18), self.cmap, discrete=6, ticks=None, nancolor=None)
                params = []
                for i_traj in selected_traj:
                    for i_it in selected_i_it[str0]:
                        params.append({
                            "X" : self.pp[varname][it0,i_traj,i_it], "Y" : self.pp["Z"][it0,i_traj,i_it]/1000, 
                            "kwargs_plt" : {"color" : cmap((i_it-it_init_in)/(6*18) + 0.5), "linewidth" : 3}, 
                            "same_ax" : i_it != selected_i_it[str0][0], "ylim":[0, 1.3], "xlim":xlim, "xlabel":xlabel, "Xname":varname, "DX_subplots" : 8, "ylabel" : "$Z$ (km)",
                            "same_fig" : i_it != selected_i_it[str0][0], "savepath" : f"{DIR}04_profile_d{self.dom.i_str}_{varname}_tc{itstr}_{str0}_itraj{i_traj}", "dpi" : 120,
                        })
                        if varname in ["U", "V", "MH", "WD180"] and varname+"2000" in self.pt:
                            params.append({
                                "X" : self.pt[varname+"2000"][it0,i_traj,i_it], "Y" : 1.28, "same" : -1, "same_ax" : True, "same_fig" : True, "style" : "o",
                            })
                fig = self.sim.plot_fig(params)

#################################################################################################################################
######  READ AND WRITE POSTPROC
#################################################################################################################################    
    
    def write_traj(self):
        self.write_postproc("X", typ="t", units="km")
        self.write_postproc("Y", typ="t", units="km")
        self.write_postproc("ZP", typ="t", units="km")
        self.write_postproc("IX", typ="t")
        self.write_postproc("IY", typ="t")
        self.write_postproc("IZ", typ="t")
        self.write_postproc("DELTA", typ="t", long_name="Time difference with the initial time", standard_name="DELTA", units="h")
    
    def load_traj(self):
        self.read_postproc(["tX", "tY", "tZP", "tIX", "tIY", "tIZ", "tDELTA"])
    
    def init_postproc(self):
        for it0 in range(self.NT0):
            filename = self.it0_to_filename(it0)
            if not os.path.exists(filename):
                with netCDF4.Dataset(filename, mode="w", format='NETCDF4_CLASSIC') as ncfile : 
                    ncfile.Title        = "py_wrf_arps LLJtraj postproc"
                    ncfile.Institution  = "LHEEA-DAUC"
                    ncfile.FMTVER       = "NetCDF 4.0 Classic"
                    ncfile.Source       = "py_wrf_arps/post/class_LLJtraj.py"
                    ncfile.References   = "See class_LLJtraj.py"
                    ncfile.Comment      = "Postproc data saved in file"
                    now = datetime.datetime.now()
                    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                    ncfile.History      = "Creation date: {date}".format(date=dt_string)
                    ncfile.createDimension('Ntraj', self.Ntraj)
                    ncfile.createDimension('NT', self.NT) 
                    ncfile.createDimension('NZ', self.NZ)
                    ncfile.createDimension('NY', self.NY)
                    ncfile.createDimension('NX', self.NX)
                    ncfile.START_DATE = self.dom.date2str(self.TIMEin[0], "WRF")
                    ncfile.COAST_DATE = self.dom.date2str(self.date0_list[it0], "WRF")
                    ncfile.END_DATE = self.dom.date2str(self.TIMEin[-1], "WRF")

        
    def write_postproc(self, varname, typ="t", long_name=None, standard_name=None, units="", latex_units=None, datatyp=np.float32):
        # Prepare
        if typ == "t":
            dims = ("Ntraj", "NT")
        elif typ == "p":
            dims = ("Ntraj", "NT", "NZ")
        else :
            raise(Exception(f"Unknown typ : {typ}"))
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
        # Write
        mode = "a"
        for it0 in range(self.NT0):
            filename = self.it0_to_filename(it0)
            if not os.path.exists(filename):
                self.init_postproc()
            ncfile = netCDF4.Dataset(filename, mode=mode, format='NETCDF4_CLASSIC')
            if mode == "a":
                if typ+varname in ncfile.variables :
                    ncfile.close()
                    ncfile = netCDF4.Dataset(filename, mode="r+", format='NETCDF4_CLASSIC')
                    ncvar = ncfile[typ+varname]
                    mode = "r+"
                    print("mode = r+ : data already exist in the file")
                else :
                    ncvar = ncfile.createVariable(typ+varname, datatyp, dims)
            if mode == "r+":
                if not typ+varname in ncfile.variables :
                    ncfile.close()
                    ncfile = netCDF4.Dataset(filename, mode="a", format='NETCDF4_CLASSIC')
                    ncvar = ncfile.createVariable(typ+varname, datatyp, dims)
                    mode = "a"
                else :
                    ncvar = ncfile[typ+varname]
            ncvar.long_name = varname
            ncvar.standard_name = varname
            # ncvar.units = units
            # ncvar.latex_units = latex_units
            if typ == "t" :
                ncvar[:] = self.pt[varname][it0]
            elif typ == "p" :
                ncvar[:] = self.pp[varname][it0]
            else :
                raise(Exception(f"Unknown typ : {typ}"))
            ncfile.close()
            
        
    def read_postproc(self, varnames=None):
        it0 = 0
        filename = self.it0_to_filename(it0)
        if os.path.exists(filename):
            ncfin = netCDF4.Dataset(filename, mode="r")
            if varnames is None :
                varnames = [v for v in ncfin.variables]
            elif type(varnames) is str :
                varnames = [varnames]
            for v in varnames :
                if v[0] == "t":
                    self.pt[v[1:]] = np.zeros((self.NT0, self.Ntraj, self.NT))
                elif v[0] == "p":
                    self.pp[v[1:]] = np.zeros((self.NT0, self.Ntraj, self.NT, self.NZ))
                else :
                    print(f"warning : Unknown typ for varname {v}")
                    varnames.remove(v)
            ncfin.close()
            for it0 in range(self.NT0):
                filename = self.it0_to_filename(it0)
                ncfin = netCDF4.Dataset(filename, mode="r")
                for v in varnames :
                    if v[0] == "t":
                        self.pt[v[1:]][it0] = ncfin[v][:]
                    elif v[0] == "p":
                        self.pp[v[1:]][it0] = ncfin[v][:]
                ncfin.close()
        else :
            print("No postproc file found")
            
    def it0_to_datestr(self, it0):
        return self.dom.date2str(self.date0_list[it0], "WRF")
    
    def it0_to_filename(self, it0) :
        date_str = self.it0_to_datestr(it0)
        return f"{self.sim.postprocdir}{self.dom.name}_LLJtraj{date_str}.nc"
        
    def filename_to_it0(self, filename) :
        date = manage_time.to_datetime(filename[-23:-4])
        return self.date0_list.index(date)
    
#################################################################################################################################
######  COMPLETE PROCEDURE
#################################################################################################################################    
    
    def compute_traj_and_save_data_for_the_first_time(self):
        print("get_data")
        self.get_traj_data()
        print("init_traj_variables")
        self.init_traj_variables()
        print("calculate_traj")
        self.calculate_traj()
        print("init_postproc")
        self.init_postproc()
        print("write_traj")
        self.write_traj()
        print("raw_to_traj")
        self.raw_to_traj(["U", "V", "Z", "LAT", "LON"])
        self.pt["MH"] = np.sqrt(self.pt["U"]**2 + self.pt["V"]**2)
        self.pt["WD180"] = manage_angle.angle180(manage_angle.UV2WD_deg(self.pt["U"], self.pt["V"]))
        print("raw_to_profile")
        self.raw_to_profile(["U", "V", "Z"])
        self.pp["MH"] = np.sqrt(self.pp["U"]**2 + self.pp["V"]**2)
        self.pp["WD180"] = manage_angle.angle180(manage_angle.UV2WD_deg(self.pp["U"], self.pp["V"]))
        print("write_postproc")
        for typ in ["t", "p"] :
            self.write_postproc("U", typ=typ)
            self.write_postproc("V", typ=typ)
            self.write_postproc("Z", typ=typ)
            self.write_postproc("MH", typ=typ)
            self.write_postproc("WD180", typ=typ, units="°")
        self.write_postproc("LAT", typ="t", long_name="Latitude", standard_name="LAT", units="°N")
        self.write_postproc("LON", typ="t", long_name="Longitude", standard_name="LON", units="°E")
        self.plot_traj()

    def get_the_traj_of_a_variable_and_save_it(self, varname, smooth=True):
        if smooth :
            self.get_sdata(varname)
        else :
            self.get_data(varname)
        self.raw_to_traj(varname)
        self.write_postproc(varname, typ="t")
        
    def get_the_profile_of_a_variable_and_save_it(self, varname, smooth=True, plot=True):
        if smooth :
            self.get_sdata(varname)
        else :
            self.get_data(varname)
        self.raw_to_profile(varname)
        self.write_postproc(varname, typ="p")
        if plot :
            self.read_postproc(["pZ"])
            self.plot_profile(varname)
            
            
    def traj_to_map_LLJ_MH(self, varname, smooth=True, plot=True):
        if smooth :
            self.get_sdata(varname)
        else :
            self.get_data(varname)
            
            
        self.raw_to_profile(varname)
        self.write_postproc(varname, typ="p")
        if plot :
            self.read_postproc(["pZ"])
            self.plot_profile(varname)