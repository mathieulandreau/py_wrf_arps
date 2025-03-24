from ..class_proj import Proj
from ..WRF_ARPS import Dom
from ..lib import manage_list

import numpy as np
import scipy
import os
# I don't understand why it didn't work with multiprocessing.Pool, I found the solution at:
# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
from multiprocessing.pool import ThreadPool as Pool

def peak_prominences_widths(row, peaks):
    prominences, left_bases, right_bases = scipy.signal.peak_prominences(row, peaks)
    widths, _, _, _ = scipy.signal.peak_widths(row, peaks, prominence_data=(prominences, left_bases, right_bases))
    return prominences, widths

def detect_LLJ(MH_in, Z_in, IZ_in, DZ_in, zaxis, max_height=500, prom_abs=2, prom_rel=0.2, width=50, squeeze=False) :
    """ Detect the presence of LLJ by searching a peak in the wind speed profile below max_height meters, and return the core speed, height, and vertical index
        Partly based on Visich, Aleksandra, et Boris Conan. 2025. https://doi.org/10.1016/j.oceaneng.2025.120749.
    Parameters
        MH_in (np.array): Horizontal wind speed array of shape (..., NZ, ...)
        Z_in (np.array): Height array of shape (..., NZ, ...)
        IZ_in (1D np.array): Vertical indices array of shape (NZ)
        DZ_in (np.array): Vertical grid spacing array of shape (..., NZ, ...)
        zaxis (int): index of the Z axis
    Optional
        max_height (float): maximum height up to which the peak is searched, default=500
        prom_abs (float): minimum absolute prominence of the peak (see scipy.signal.find_peak), default=2 (see Visich and Conan, step 2)
        prom_rel (float): minimum prominence relative to the peak velocity (see scipy.signal.find_peak), default=0.2 (see Visich and Conan, step 2)
        width (float): minimum width of the peak in meters (see scipy.signal.find_peak), default=50
    Return
        LLJ (np.array): Status of the detection (0=no LLJ, 1=LLJ, 2=several peaks, 3=peak with small width, 4=peak with small prominence, 5=peak at higher altitude), shape=(..., 1, ...)
        LLJ_IZ (np.array): Vertical index of the LLJ core, shape=(..., 1, ...)
        LLJ_Z (np.array): LLJ core height, shape=(..., 1, ...)
        LLJ_MH (np.array): LLJ core speed, shape=(..., 1, ...)
        LLJ_PROM (np.array): Prominence of the peak found, shape=(..., 1, ...)
        LLJ_WIDTH (np.array): Width of the peak found, shape=(..., 1, ...)
    20/03/2025 : Mathieu Landreau
    """  
    # 
    DZ0 = np.take(DZ_in, 0, axis=zaxis)
    shape = DZ0.shape
    MH = np.insert(MH_in, 0, np.zeros(shape), axis=zaxis)
    Z = np.insert(Z_in, 0, np.zeros(shape), axis=zaxis)
    DZ = np.insert(DZ_in, 0, DZ0, axis=zaxis)
    IZ = np.append(IZ_in, np.nanmax(IZ_in)+1)
    
    # Roughly step 1 and 2 from Visich and Conan, 2025
    temp = np.apply_along_axis(peak_prominences_widths, zaxis, MH, IZ)
    PROM, WIDTH = np.take(temp, 0, axis=zaxis), np.take(temp, 1, axis=zaxis)
    WIDTH = WIDTH*DZ
    m1 = 1*np.logical_or(PROM>=prom_abs, np.logical_and(PROM>=prom_rel*MH, PROM>=1)) # Roughly equivalent to their step 2
    m2 = 1*(WIDTH>=width)
    m3 = 1*(Z<=max_height)
    m4 = 1*(Z>max_height)*(Z<=max(2*max_height, 1000))
    LLJ = 1*(np.sum(m1*m2*m3, axis=zaxis, keepdims=True)==1) #there is only one valid peak
    LLJ[np.logical_and(LLJ==0, np.sum(m1*m2*m3, axis=zaxis, keepdims=True)>1)] = 2 #there are several valid peaks
    LLJ[np.logical_and(LLJ==0, np.sum(m1*m3, axis=zaxis, keepdims=True)>0)] = 3 #there is a valid peak with a too small width
    LLJ[np.logical_and(LLJ==0, np.sum(m2*m3, axis=zaxis, keepdims=True)>0)] = 4 #there is a valid peak with a too small prominence
    LLJ[np.logical_and(LLJ==0, np.sum(m1*m2*m4, axis=zaxis, keepdims=True)>0)] = 5 #there is a valid peak at higher altitude
    LLJ = LLJ.astype(int)
    
    LLJ_IZ = np.expand_dims(np.nanargmax(m1*m2*m3, axis=zaxis), axis=zaxis)
    LLJ_Z = np.take_along_axis(Z, LLJ_IZ, axis=zaxis)
    LLJ_MH = np.take_along_axis(MH, LLJ_IZ, axis=zaxis)
    LLJ_PROM = np.take_along_axis(PROM, LLJ_IZ, axis=zaxis)
    LLJ_WIDTH = np.take_along_axis(WIDTH, LLJ_IZ, axis=zaxis)
    LLJ_IZ[LLJ != 1] = -1
    LLJ_Z[LLJ != 1] = LLJ_MH[LLJ != 1] = LLJ_PROM[LLJ != 1] = LLJ_WIDTH[LLJ != 1] = np.nan
    LLJ_IZ = LLJ_IZ-1
    if squeeze :
        LLJ = np.squeeze(LLJ)
        LLJ_IZ = np.squeeze(LLJ_IZ)
        LLJ_MH = np.squeeze(LLJ_MH)
        LLJ_Z = np.squeeze(LLJ_Z)
        LLJ_PROM = np.squeeze(LLJ_PROM)
        LLJ_WIDTH = np.squeeze(LLJ_WIDTH)
    return LLJ, LLJ_IZ, LLJ_Z, LLJ_MH, LLJ_PROM, LLJ_WIDTH

class LLJ(): 
    def __init__(self, sim):
        """ compute LLJ characteristics and save in post proc files
        Parameters
            self (LLJ)
            sim: a class_Proj object
        20/03/2025 : Mathieu Landreau
        """  
        self.sim = sim
        
    def detect_LLJ(self, dom, max_height=500, prom_abs=2, prom_rel=0.2, width=50, DX_smooth_rel=10, DX_smooth=None, **kw_get):
        """ Get data and call detect_LLJ (defined outside the class)
        Parameters
            self (LLJ)
            dom: see classProj.Proj.get_dom
        Optional
            max_height, prom_abs, prom_rel, width: see detect_LLJ
            DX_smooth_rel (float): if DX_smooth is None, it takes the value of DX_smooth_rel*DX_KM
            kw_get (dict): see Dom.get_data
        20/03/2025 : Mathieu Landreau
        """ 
        if DX_smooth is None and DX_smooth_rel is not None:
            DX_KM = self.sim.get_data(dom, "DX")/1000
            DX_smooth = DX_smooth_rel*DX_KM
        MH, Z, IZ, DZ = self.sim.get_data(dom, ["MH", "Z", "iz", "DZ"], DX_smooth=DX_smooth, **kw_get)
        zaxis = self.sim.get_dom(dom).find_axis("z", varname="MH", **kw_get)
        return detect_LLJ(MH, Z, IZ, DZ, zaxis, max_height, prom_abs, prom_rel, width)
    
    def write_postproc_1time(self, it, dom, dry, crop, kw):
        print(it, end=" ")
        if not dry :
            LLJ, LLJ_IZ, LLJ_Z, LLJ_MH, LLJ_PROM, LLJ_WIDTH = self.detect_LLJ(dom, itime=it, crop=crop, **kw)
            dom.write_postproc("LLJ", LLJ, ('y', 'x'), itime=it, long_name="LLJ detection", standard_name="LLJ", units="", latex_units="", typ=np.int64)  
            dom.write_postproc("LLJ_IZ", LLJ_IZ, ('y', 'x'), itime=it, long_name="LLJ core index", standard_name="LLJ_IZ", units="", latex_units="", typ=np.int64) 
            dom.write_postproc("LLJ_Z", LLJ_Z, ('y', 'x'), itime=it, long_name="LLJ core height", standard_name="LLJ_Z", units="m", latex_units="m", typ=np.float32) 
            dom.write_postproc("LLJ_MH", LLJ_MH, ('y', 'x'), itime=it, long_name="LLJ core speed", standard_name="LLJ_MH", units="m.s-1", latex_units="m.s^{-1}", typ=np.float32) 
            dom.write_postproc("LLJ_PROM", LLJ_PROM, ('y', 'x'), itime=it, long_name="LLJ peak prominence", standard_name="LLJ_PROM", units="m.s-1", latex_units="m.s^{-1}", typ=np.float32) 
            dom.write_postproc("LLJ_WIDTH", LLJ_WIDTH, ('y', 'x'), itime=it, long_name="LLJ peak width", standard_name="LLJ_WIDTH", units="m", latex_units="m", typ=np.float32) 
                
    
    def write_postproc(self, dom, itime="ALL_TIMES", nprocs=1, crop=None, max_height=500, dry=False, **kw):
        """ Save the results in the postproc file
        Parameters
            self (LLJ)
        Optional
            savepath (str): path to save the pickle file
        04/03/2025 : Mathieu Landreau
        """ 
        dom = self.sim.get_dom(dom)
        if dom.software in ["AROME"]: raise(Exception("Cannot write postproc with this kind of domain: ", dom.software))
        if crop is None:
            izmax = dom.nearest_z_index(1.2*max_height)
            crop = ([0, izmax], "ALL", "ALL")
        IT, TIME = dom.get_data(["it", "TIME"], itime=itime)
        
        # self.temp_args = (dom, dry, crop, kw)
        if not manage_list.is_iterable(IT):
            IT = [IT]
        NT = len(IT)
        if nprocs > 1 :
            inputs = [(it, dom, dry, crop, kw) for it in IT]
            with Pool(processes=nprocs) as pool:
                pool.starmap(self.write_postproc_1time, inputs)
        else :
            for it in IT :
                self.write_postproc_1time(it, dom, dry, crop, kw)
    