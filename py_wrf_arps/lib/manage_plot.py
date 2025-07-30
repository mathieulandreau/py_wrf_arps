from ..default_params import default_params
from ..lib import manage_angle, manage_dict, manage_time, manage_list

import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as mpl_Rectangle
from matplotlib.colors import LinearSegmentedColormap
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.dates as mdates
import matplotlib
import matplotlib.animation
from multiprocessing import Pool, cpu_count
import numpy as np
import math
import datetime
import os

width_screen = 35

#Taken from https://stackoverflow.com/questions/32736503/cyclically-shifting-a-colormap
class roll_cmap(LinearSegmentedColormap):
    def __init__(self, cmap, shift, N=None):
        assert 0. < shift < 1.
        self.shift = shift
        self.cmap = plt.get_cmap(cmap, N)
        self.N = cmap.N
        self.monochrome = self.cmap.monochrome
        self._x = np.linspace(0.0, 1.0, 255)
        self._y = np.roll(np.linspace(0.0, 1.0, 255),int(255.*shift))
    def __call__(self, xi, alpha=1.0, **kw):
        """
        if abs(xi - self.shift*255) < 1 :
            yi = 1.0
        else :
        
        yi = np.interp(xi, self._x, self._y)
        yi[np.abs(xi - self.shift*255) < 1] = 1.0
        """
        
        yi = (xi + self.shift) % 1
        return self.cmap(yi, alpha)

# Useless : just type _r at the end of the colormap (Spectral_r, bwr_r, ...)
class inversed_cmap(LinearSegmentedColormap):
    def __init__(self, cmap, N=None):
        self.cmap = plt.get_cmap(cmap, N)
        self.N = 255 if N == None else N
        self.monochrome = self.cmap.monochrome
        self._x = np.linspace(0.0, 1.0, self.N)
        self._y = self._x[::-1]
    def __call__(self, xi, alpha=1.0, **kw):
        yi = 1-xi
        return self.cmap(yi, alpha)

warnings.filterwarnings("ignore", module="matplotlib\..*") #disable matplotlib warnings
warnings.filterwarnings("ignore") #disable all warnings
plt.rc('font', size=20) #controls default text size
plt.rc('axes', titlesize=20) #fontsize of the title
plt.rc('axes', labelsize=20) #fontsize of the x and y labels
plt.rc('xtick', labelsize=20) #fontsize of the x tick labels
plt.rc('ytick', labelsize=20) #fontsize of the y tick labels
plt.rc('legend', fontsize=20) #fontsize of the legend
dtFmt = mdates.DateFormatter('%d%b %Hh')

cmap_circ = plt.cm.twilight
angle_mer = 135
shift = ((360-angle_mer)/360)%1
twilight_rolled = roll_cmap(cmap_circ, shift=shift)

colors = list(matplotlib.rcParams["axes.prop_cycle"])
for i, c in enumerate(colors) :
    colors[i] = c["color"]
colors2 = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'gray', 'lime']
markers = ["o", "v", "^", ">", "<", "s", "*"]
linestyles = [(0, ()),                   # solid
              (0, (5, 1)),               # densely dashed
              (0, (1, .5)),              # densely dotted
              (0, (3, 1, 1, 1)),         # densely dashdotted
              (0, (3, 1, 1, 1, 1, 1)),   # densely dashdotted
              (0, (5, 3)),               # loosely dashed
              (0, (2, 1)),               # loosely dotted
              (0, (3, 3, 1, 3)),         # loosely dashdotted
              (0, (3, 3, 1, 3, 1, 3))]   # loosely dashdotdotted
VARPLOT = {
    "fps" : 1,
}
 
tab_cmap = {
    0 : "Spectral_r",
    1 : "terrain", 
    2 : twilight_rolled, # "twilight", "twilight_shifted",
    3 : "Spectral_r",#"spectral", # "coolwarm", "bwr",
    4 : "tab20",
    5 : "tab20",
    6 : "tab20",
    7 : "Greys_r", #for Landmask
    8 : "bwr_r",
    9 : "bwr",
}
    
FLAGS = {
    'verbose': True,
    'warning': True,
    'conc':False
}


debug = False

def plot_fig(params_list, n_procs=1):
    if type(params_list) is not list : params_list = [params_list]
    N_figures = 1
    for i_params, params in enumerate(params_list) :
        if i_params > 0 and "same_fig" in params and not params["same_fig"] :
            N_figures += 1
        params["i_fig"] = N_figures
            
    if N_figures > 1 :
        n_procs = min(N_figures, n_procs)
        params_list_list = []
        for i_fig in range(N_figures) :
            temp_list = []
            i_temp = 0
            fig = []
            while i_temp < len(params_list) :
                if params_list[i_temp]["i_fig"] == i_fig+1:
                    temp_list.append(params_list[i_temp])
                i_temp += 1
            if n_procs == 1 :
                fig.append(plot_fig(temp_list))
            else :
                params_list_list.append((temp_list,))
        if n_procs > 1 :
            try :
                with Pool(processes=n_procs) as pool:
                    fig = pool.starmap(plot_fig, params_list_list)     
            except :
                print("warning : no parallel")
                fig = []
                for temp_list in params_list_list :
                    fig.append(plot_fig(temp_list[0]))
        return fig
        
    
    # IF ONLY ONE FIGURE :
    N_subplots = 0
    DX_subplots, DY_subplots = 12, 8
    NX_subplots = 2
    dpi = 60
    for i_params, params in enumerate(params_list) :
        if not("same_ax" in params) or not(params["same_ax"]) or i_params==0 :
            N_subplots += 1
        params["i_ax"] = N_subplots
        if "DX_subplots" in params : DX_subplots = params["DX_subplots"]
        if "DY_subplots" in params : DY_subplots = params["DY_subplots"]
        if "NX_subplots" in params : NX_subplots = params["NX_subplots"]
        if "dpi" in params : dpi = params["dpi"]
    if debug : manage_dict.print_dict(params_list)
    NX_subplots = min(NX_subplots, N_subplots)
    is_video = False
    tab_proj = N_subplots*[None]
    for params in params_list : 
        if "video" in params and params["video"] != False :
            is_video = params["video"]
        if params["typ"] in ["UV", "UV2", "WDH"] :
            tab_proj[params["i_ax"]-1] = "polar"
    
    savepath, savefmt, bbox_inches = get_savepath_fmt_bbox(params_list, is_video)
    print("ploting fig ", savepath)
    NY_subplots = (N_subplots-1)//NX_subplots + 1
    if NX_subplots*DX_subplots > width_screen :
        fac = DY_subplots/DX_subplots
        DX_subplots = width_screen/NX_subplots
        DY_subplots = fac*DX_subplots
    fig = plt.figure(figsize = [NX_subplots*DX_subplots, NY_subplots*DY_subplots], dpi=dpi)
    ax_list = []
    for i_ax in range(N_subplots) :
        ax_list.append(fig.add_subplot(NY_subplots, NX_subplots, i_ax+1, projection=tab_proj[i_ax])) 
    if is_video != False :
        NT, fps = get_NT_fps(params_list)
        #plot the first time
        for params in params_list :
            i_ax = params["i_ax"]-1
            ax = ax_list[i_ax]
            params["plot_obj"] = ax_plot(ax, it=0, **params)
        #create the animate function
        def animate(it) :
            if it%20 == 0 :
                print("animate", it)
            for params in params_list :
                i_ax = params["i_ax"]-1
                ax = ax_list[i_ax]
                if "animate" in params and not params["animate"] :
                    continue
                if "plot_obj" in params :
                    params["plot_obj"] = ax_plot(ax, it=it, **params)
        if is_video == "frame" : # save each frame separately
            print(NT, "frames")
            TIME = None
            for params in params_list :
                if params["typ"] == "DATE" :
                    TIME = params["Z"]
            if TIME is not None :
                tstr = manage_time.date_to_str(TIME, fmt="%Y-%m-%d_%H-%M-%S")
            else :
                tstr = np.arange(NT).astype(str)
            for it in range(NT) :
                animate(it)
                fig.savefig(savepath+"_"+tstr[it]+"."+savefmt)
        else : # real video
            print(NT)
            anim = matplotlib.animation.FuncAnimation(fig, animate, interval=100, frames=NT-1)
            if savepath is not None :
                writervideo = matplotlib.animation.FFMpegWriter(fps=fps) 
                anim.save(savepath+"."+savefmt, writer=writervideo)
            return fig
            
    else :
        for params in params_list :
            i_ax = params["i_ax"]-1
            ax = ax_list[i_ax]
            ax_plot(ax, **params)
        if savepath is not None :
            #print("saving fig at path : "+savepath)
            fig.savefig(savepath+"."+savefmt, format=savefmt, bbox_inches=bbox_inches)                
        return fig

def get_savepath_fmt_bbox(params_list, video) :
    #default values
    savepath = None 
    if video == True :
        savefmt = "avi"
    else :
        savefmt = "png"
    #get from params_list  
    bbox_inches = False
    for params in params_list :
        if "savepath" in params :
            savepath = params["savepath"]
        if "savefmt" in params : 
            savefmt = params["savefmt"]
        if "tight" in params and params["tight"] :
            bbox_inches = "tight"
    if savepath is not None and video == "frame":
        basename = os.path.basename(savepath)
        savepath = savepath+"/"+basename
    if savepath is not None and not os.path.exists(os.path.dirname(savepath)):
        print('Create figure directory in {0}'.format(os.path.dirname(savepath)))
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
    return savepath, savefmt, bbox_inches

def get_NT_fps(params_list):
    NT = 0
    fps = VARPLOT["fps"]
    for params in params_list :
        if "NT" in params :
            NT = params["NT"]
        if "fps" in params :
            fps = params["fps"]
    if NT < 1 :
        print("error in manage_plot.plot_fig : NT is undefined for this video")
        raise
    return NT, fps

def ax_plot(ax, **params):
    typename = params["typ"].upper()  
    #manage_dict.print_dict(params, typename)
    if typename in ["2D_HORIZONTAL" , "2DH", "2D", "ZT", "VCROSS", "VERTCROSS", "2DV", "2D_VERTICAL"] or "2D" in typename:
        plot_obj = ax_2D_plot(ax, **params)
    elif typename in ["TIME", "T", "1DT", "1D", "POINT"]:
        plot_obj = ax_1D_plot(ax, **params)
    elif typename == "SCATTER":
        plot_obj = ax_scatter(ax, **params)
    elif typename == "QUIVER":
        plot_obj = ax_quiver(ax, **params)
    elif typename == "BARBS":
        plot_obj = ax_barbs(ax, **params)
    elif typename == "UV":
        plot_obj = ax_hodograph(ax, **params)
    elif typename == "UV2":
        plot_obj = ax_hodograph2(ax, **params)
    elif typename == "RECTANGLE":
        plot_obj = ax_rectangle(ax, **params)
    elif typename == "LINE":
        plot_obj = ax_line(ax, **params)
    elif typename in ["CONTOUR", "LON", "LAT", "LANDMASK", "MASK"]:
        plot_obj = ax_contour(ax, **params)
    elif typename == "NIGHTTIME":
        plot_obj = ax_nighttime(ax, **params)
    elif typename == "LANDSEA":
        plot_obj = ax_landsea(ax, **params)
    elif typename == "WDH":
        plot_obj = ax_WD_hist(ax, **params)
    elif typename in ["DATE", "TEXT"]:
        plot_obj = ax_date(ax, **params)
    elif typename == "AXVLINE":
        plot_obj = ax_vline(ax, **params)
    elif typename == "AXHLINE":
        plot_obj = ax_hline(ax, **params)
    elif typename == "AXVSPAN":
        plot_obj = ax_vspan(ax, **params)
    elif typename == "AXHSPAN":
        plot_obj = ax_hspan(ax, **params)
    else : 
        print("error in manage_plot.ax_plot : unknown type of plot ("+typename+"), cannot plot anything")
    if not "plot_obj" in params :
        if "title" in params : ax.set_title(params["title"])
        if "xlabel" in params : ax.set_xlabel(params["xlabel"])
        if "ylabel" in params : ax.set_ylabel(params["ylabel"])
        if "xscale" in params : ax.set_xscale(params["xscale"])
        if "yscale" in params : ax.set_yscale(params["yscale"])
        if "yticklabels" in params : ax.set_yticklabels(params["yticklabels"])
        if "xticklabels" in params : ax.set_xticklabels(params["xticklabels"])
        if "xticks" in params : 
            if manage_list.is_iterable(params["xticks"]) : ax.set_xticks(params["xticks"])
            else : ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(params["xticks"]))
        if "yticks" in params : 
            if manage_list.is_iterable(params["yticks"]) : ax.set_yticks(params["yticks"])
            else : ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(params["yticks"]))
        if "minorxticks" in params : 
            if manage_list.is_iterable(params["minorxticks"]) : ax.set_xticks(params["minorxticks"], minor=True)
            else : ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(params["minorxticks"]))
        if "minoryticks" in params : 
            if manage_list.is_iterable(params["minoryticks"]) : ax.set_yticks(params["minoryticks"], minor=True)
            else : ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(params["minoryticks"]))
        if "label" in params :
            loc = manage_dict.getp("legend_loc", params, "best")
            ax.legend(loc=loc)
        if "grid" in params :
            grid_axis = manage_dict.getp("grid_axis", params, default="both")
            grid_which = manage_dict.getp("grid_which", params, default="major")
            ax.grid(params["grid"], axis=grid_axis, which=grid_which)
            if grid_which == "both" :
                ax.grid(params["grid"], axis=grid_axis, which="major")
                ax.grid(params["grid"], axis=grid_axis, which="minor", alpha=0.3)
            else :
                ax.grid(params["grid"], axis=grid_axis, which=grid_which)
        if "xlim" in params : ax.set_xlim(params["xlim"])
        if "ylim" in params : ax.set_ylim(params["ylim"])
                
    return plot_obj

def ax_1D_plot(ax, X, Y, style='-', label=None, plot_obj=None, it=None, animate=False, kwargs_plt={}, **kwargs):
    if it is not None and animate : # for video
        Xi = X[it]
        Yi = Y[it]
    else :
        Xi = X
        Yi = Y
    if plot_obj is not None :
        plot_obj.set_data(np.squeeze(Xi), np.squeeze(Yi))
    else :
        plot_obj, = ax.plot(np.squeeze(Xi), np.squeeze(Yi), style, label=label, **kwargs_plt)
        if it is not None and not "xlim" in kwargs:
            xmin, xmax = plt.xlim()
            xmax = max(np.nanmax(X), xmax)
            xmin = min(np.nanmin(X), xmin)
            ax.set_xlim([xmin, xmax])
        if it is not None and not "ylim" in kwargs:
            ymin, ymax = plt.ylim()
            ymax = max(np.nanmax(Y), ymax)
            ymin = min(np.nanmin(Y), ymin)
            ax.set_ylim([ymin, ymax])
    return plot_obj
    
def ax_scatter(ax, X, Y, Z, clim=None, clabel=None, label=None, it=None, ticks=None, ticklabels=None, nancolor=None, plot_obj=None, aspect=False, plot_cbar=True, discrete=None, kwargs_plt={}, typ="SCATTER", **kwargs):
    """
    make a pcolor plot with the params
    ax : an ax or subplot of a pyplot figure (see plot_fig), passed as a pointer
    params : a dict with the parameters of the plot containing
        (mandatory)
        "X", "Y" : 2D arrays of shape (NY, NX) : the meshgrid domain
        "Z" : 2D arrays of shape (NY, NX), The color to plot
        (optional)
        "kwargs_plt" : the kwargs of pcolor
        "clim" : an array or list of length 2,  colorbar limit
        "clabel" : label of colorbar (corresponding to Z value)
    """
    cmap = kwargs_plt["cmap"] if "cmap" in kwargs_plt else None
    Z = np.squeeze(Z)
    Y = np.squeeze(Y)
    X = np.squeeze(X)
    if it is not None :
        Zi = Z[it]
        if X.ndim == 3 :
            Xi = X[it]
        else :
            Xi = X
        if Y.ndim == 3 :
            Yi = Y[it] 
        else : 
            Yi = Y 
    else :
        Zi = Z
        Xi = X
        Yi = Y
    
    if plot_obj is not None :
        if debug : print("updating scatter", it)
        plot_obj.set_array(Zi.flatten())
    else :
        if debug : print("printing scatter")
        vmin, vmax, extend, cmap, ticks = get_cmap_extend(clim, Z, cmap, discrete, ticks, nancolor)
        kwargs_plt["cmap"] = cmap
        if aspect : ax.set_aspect('equal', adjustable='box')
        plot_obj = ax.scatter(Xi, Yi, c=Zi, vmin=vmin, vmax=vmax, label=label, **kwargs_plt)
        if plot_cbar : 
            cbar = plt.colorbar(mappable=plot_obj, ax=ax, extend=extend, ticks=ticks)
            cbar.set_label(clabel)
            if ticklabels is not None :
                cbar.ax.set_yticklabels(ticklabels)
    return plot_obj

def ax_2D_plot(ax, X, Y, Z, clim=None, ticks=None, ticklabels=None, clabel=None, it=None, plot_obj=None, plot_cbar=True, discrete=None, nancolor=None, 
               mesh=True, kwargs_plt={}, typ="2D", **kwargs):
    """
    make a pcolor plot with the params
    ax : an ax or subplot of a pyplot figure (see plot_fig), passed as a pointer
    params : a dict with the parameters of the plot containing
        (mandatory)
        "X", "Y" : 2D arrays of shape (NY, NX) : the meshgrid domain
        "Z" : 2D arrays of shape (NY, NX), The color to plot
        (optional)
        "kwargs_plt" : the kwargs of pcolor
        "clim" : an array or list of length 2,  colorbar limit
        "clabel" : label of colorbar (corresponding to Z value)
    """
    cmap = kwargs_plt["cmap"] if "cmap" in kwargs_plt else None
    Z = np.squeeze(Z)
    Y = np.squeeze(Y)
    X = np.squeeze(X)
    if it is not None :
        Zi = Z[it]
        if X.ndim == 3 :
            Xi = X[it]
        else :
            Xi = X
        if Y.ndim == 3 :
            Yi = Y[it] 
        else : 
            Yi = Y 
    else :
        Zi = Z
        Xi = X
        Yi = Y
    #nans in X or Y destroy the plot
    if not type(Xi[0][0]) in [datetime.datetime] and np.any(np.isnan(Xi)) :
        pos = np.isnan(Xi)
        Zi[pos] = np.nan
        Xi[pos] = np.nanmin(Xi[np.logical_not(pos)])
    if not type(Yi[0][0]) in [datetime.datetime] and np.any(np.isnan(Yi)) :
        pos = np.isnan(Yi)
        Zi[pos] = np.nan
        Yi[pos] = np.nanmin(Yi[np.logical_not(pos)])
    
    if plot_obj is not None :
        if debug : print("updating pcolor", it)
        plot_obj.set_array(Zi.flatten())
    else :
        if debug : print("printing pcolor")
        vmin, vmax, extend, cmap, ticks = get_cmap_extend(clim, Z, cmap=cmap, discrete=discrete, ticks=ticks, nancolor=nancolor)
        kwargs_plt["cmap"] = cmap
        if typ.upper() in ["2DH", "2D_HORIZONTAL"] : ax.set_aspect('equal', adjustable='box')
        shading = "gouraud" if not mesh else None
        plot_obj = ax.pcolormesh(Xi, Yi, Zi, vmin=vmin, vmax=vmax, shading=shading, **kwargs_plt)
        if plot_cbar : 
            cbar = plt.colorbar(mappable=plot_obj, ax=ax, extend=extend, ticks=ticks)
            cbar.set_label(clabel)
            if ticklabels is not None :
                cbar.ax.set_yticklabels(ticklabels)
    return plot_obj
            
def ax_contour(ax, X, Y, Z, clabel=True, label=None, fontsize=15, it=None, plot_obj=None, kwargs_plt=default_params["CONTOUR"]["kwargs_plt"], **kwargs):
    """
    Description
        make a contour plot with the params
    Parameters
        ax : an ax or subplot of a pyplot figure (see plot_fig), passed as a pointer
        X, Y : 2D arrays of shape (NY, NX) : the meshgrid domain
        Z : 2D arrays of shape (NY, NX) : The array to compute contours
    Optional
        label : str, a label to add to the legend
        clabel : boolean : True to print the values inside each contour, default is False
    """
    if it is not None :
        if Z.ndim == 3 :
            Zi = Z[it]
        else :
            Zi = Z
        if X.ndim == 3 :
            Xi = X[it]
            Yi = Y[it]  
        else :
            Xi = X
            Yi = Y 
    else :
        Zi = Z
        Xi = X
        Yi = Y
    Zi = np.squeeze(Zi)
    Yi = np.squeeze(Yi)
    Xi = np.squeeze(Xi)
    if plot_obj is not None : 
        for old_c in plot_obj[0].collections :
            try : 
                if debug : print("remove old_c")
                old_c.remove() # I didn't find any way to animate the existing contour, so I delete it and redraw it
            except :
                if debug : print("cannot remove old_c")
        if plot_obj[1] is not None :
            for old_l in plot_obj[1] :
                try : 
                    old_l.remove()
                except :
                    if debug : print("cannot remove old_l")
    plot_obj = [None, None]
    plot_obj[0] = ax.contour(Xi, Yi, Zi, **kwargs_plt)
    if clabel : 
        plot_obj[1] = ax.clabel(plot_obj[0], plot_obj[0].levels, inline=True, fontsize=fontsize)
    if label is not None :
        ax.plot(0, 0, kwargs_plt["colors"], label=label)
    return plot_obj

def ax_line(ax, x1, x2, y1, y2, kwargs_plt=default_params["LINE"]["kwargs_plt"], kwargs_1=default_params["LINE"]["kwargs_1"], **kwargs):
    """
    Draw a line with the params
    x1, x2, y1, y2 : float : coordinates of the two points
    (optional)
    kwargs_plt : the kwargs of plot
    kwargs_1 : the kwargs of the 1st point
    """
    ax.plot([x1, x2], [y1, y2], **kwargs_plt)
    ax.plot(x1, y1, **kwargs_1)
        
def ax_rectangle(ax, xy, width, height, rlabel="", kwargs_plt=default_params["RECTANGLE"]["kwargs_plt"], **kwargs):
    """
    Draw a rectangle with the params
    ax : an ax or subplot of a pyplot figure (see plot_fig), passed as a pointer
    xy : list of 2 floats : (x1, y1) : position of the lower left corner
    width, height : float 
    (optional)
    kwargs_plt : the kwargs of add_patch
    rlabel : a label to print in the rectangle
    """
    ax.add_patch(mpl_Rectangle(xy, width, height, **kwargs_plt))
    ax.text(xy[0], xy[1], rlabel)


def ax_quiver(ax, X, Y, U, V, size=1, step=18, stepx=None, label=None, plot_legend=True, it=None, plot_obj=None, kwargs_plt=default_params["QUIVER"]["kwargs_plt"], args_key=default_params["QUIVER"]["args_key"], **kwargs) :
    """
    make a quiver plot with the params
    ax : an ax or subplot of a pyplot figure (see plot_fig), passed as a pointer
    X, Y : 2D arrays of shape (NY, NX) : the meshgrid domain
    U, V : 2D arrays of shape (NY, NX) : the coordinates of the vectors
    (optional)
    kwargs_plt : the kwargs of quiver
    size : a coefficient to apply on arrows width default is 1
    """
    if it is not None :
        Ui = U[it]
        Vi = V[it]
    else :
        Ui = U
        Vi = V
    # Ui = np.squeeze(Ui)
    # Vi = np.squeeze(Vi)
    
    if X.ndim == 3 :
        Xi = X[it]
    else :
        Xi = X
    if Y.ndim == 3 :
        Yi = Y[it] 
    else : 
        Yi = Y 
        
    kwargs_plt["width"]*= size
    kwargs_plt["scale"]*= args_key[0]/5 / size
    NY, NX = Xi.shape
    if stepx is None :
        N = max(NX, NY)
    else :
        N = NY
    quiver_step = max(N//step, 1)
    sy = slice(quiver_step//2, N, quiver_step)
    if stepx is None :
        sx = sy
    else :
        quiver_step = max(NX//stepx, 1)
        sx = slice(quiver_step//2, NX, quiver_step)
    s = (sy, sx)
    if plot_obj is not None : 
        plot_obj.set_UVC(Ui[s], Vi[s])
    else :
        plot_obj = ax.quiver(Xi[s], Yi[s], Ui[s], Vi[s], label=label, **kwargs_plt)
        if plot_legend : 
            xlim = ax.set_xlim()
            ylim = ax.set_ylim()
            ix0 = 0.0
            iy0 = 0.0
            x0 = xlim[0] * (1-ix0) + xlim[1] * ix0
            y0 = ylim[0] * (1-iy0) + ylim[1] * iy0
            dx = (xlim[1] - xlim[0])*0.2*(0.6+0.4*size)
            dy = (ylim[1] - ylim[0])*0.05
            ax.add_patch(matplotlib.patches.Rectangle([x0, y0], dx, dy, color=[0.8, 0.8, 0.8, 0.9]))
            matplotlib.pyplot.quiverkey(plot_obj, x0+dx/2.5, y0+dy/2, *args_key, coordinates = "data", labelpos="E")
    return plot_obj
 

def ax_barbs(ax, X, Y, U, V, size=1, step=18, stepx=None, label=None, it=None, plot_obj=None, kwargs_plt=default_params["BARBS"]["kwargs_plt"], **kwargs) :
    """
    make a barbs plot with the params
    ax : an ax or subplot of a pyplot figure (see plot_fig), passed as a pointer
    X, Y : 2D arrays of shape (NY, NX) : the meshgrid domain
    U, V : 2D arrays of shape (NY, NX) : the coordinates of the vectors
    (optional)
    kwargs_plt : the kwargs of barbs
    size : a coefficient to apply on arrows width default is 1
    """
    if it is not None :
        Ui = U[it]
        Vi = V[it]
    else :
        Ui = U
        Vi = V
    Ui = np.squeeze(Ui)
    Vi = np.squeeze(Vi)
    
    if X.ndim == 3 :
        Xi = X[it]
    else :
        Xi = X
    if Y.ndim == 3 :
        Yi = Y[it] 
    else : 
        Yi = Y 
        
    kwargs_plt["width"]*= size
    NY, NX = Xi.shape
    if stepx is None :
        N = max(NX, NY)
    else :
        N = NY
    barbs_step = max(N//step, 1)
    sy = slice(barbs_step//2, N, barbs_step)
    if stepx is None :
        sx = sy
    else :
        barbs_step = max(NX//stepx, 1)
        sx = slice(barbs_step//2, NX, barbs_step)
    s = (sy, sx)
    if plot_obj is not None : 
        plot_obj.set_UVC(Ui[s], Vi[s])
    else :
        plot_obj = ax.barbs(Xi[s], Yi[s], Ui[s], Vi[s], label=label, **kwargs_plt)
    return plot_obj
    
def ax_hodograph(ax, U, V, Z=None, Z_PBL=None, style='-', label=None, cmap="Spectral_r", ticks=None, discrete=None, scatter=True, clim=None, it=None, NZ=15, clabel=None, plot_obj=None, kwargs_plt=default_params["UV"]["kwargs_plt"], **kwargs):
    if it is not None :
        Ui = U[it]
        Vi = V[it]
        if Z is not None and Z.ndim == 2 :
            Zi = Z[it]
        else : 
            Zi = Z
        if Z_PBL is not None and type(Z_PBL) in [list, np.array, np.ndarray] :
            Z_PBLi = Z_PBL[it]
        else :
            Z_PBLi = Z_PBL
    else :
        Ui = U
        Vi = V
        Zi = Z
        if Z_PBL is not None :
            Z_PBLi = Z_PBL
    MH = np.sqrt(U**2 + V**2)
    WD_rad = manage_angle.UV2WD_rad(U, V)

    if plot_obj is not None :
        plot_obj[0].set_data(WD_rad, MH)
        if Z is not None :
            plot_obj[1].set_offsets(np.transpose(np.array([WD_rad, MH])))
            if Z_PBL is not None :
                plot_obj[2].set_ticks(Zi, labels=Zi)
    else :
        plot_obj = [None, None, None]
        # Create a hodograph plot
        #ax.set_projection("polar")
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(90)
        #ax.set_yticklabels([])
        plot_obj[0], = ax.plot(WD_rad, MH, style, label=label, **kwargs_plt)
    
        if Z is not None :
            vmin, vmax, extend, cmap, ticks = get_cmap_extend(clim, Z, cmap, discrete, ticks)
            if Z_PBL is not None :
                WD_PBL = np.interp(Z_PBLi, Zi, WD_rad)
                MH_PBL = np.interp(Z_PBLi, Zi, MH)
                ax.plot(WD_PBL, MH_PBL, "or")
            if scatter : 
                DNZ = len(Zi)//NZ + 1
                s = slice(0, -1, DNZ)
                plot_obj[1] = ax.scatter(WD_rad[s], MH[s], c=Zi[s], s=100, lw=0.5, edgecolors="k", cmap=cmap)
                plot_obj[1].set_clim(vmin=vmin, vmax=vmax)
                plot_obj[2] = plt.colorbar(ticks=Zi[s], mappable=plot_obj[1], ax=ax, extend=extend)
                plot_obj[2].set_label(clabel)
                if Z_PBL is not None :
                    plot_obj[2].ax.plot([0, 1], [Z_PBLi, Z_PBLi], "r")
            
    return plot_obj

def ax_hodograph2(ax, U, V, NZ=10, style='-', label=None, color=None, it=None, plot_obj=None, \
                 kwargs_plt=default_params["UV2"]["kwargs_plt"], **kwargs):
    if it is not None :
        Ui = U[it]
        Vi = V[it]
    else :
        Ui = U
        Vi = V
    MH = np.sqrt(U**2 + V**2)
    WD_rad = manage_angle.UV2WD_rad(U, V)

    if plot_obj is not None :
        plot_obj[0].set_data(WD_rad, MH)
        plot_obj[1].set_data(WD_rad[0], MH[0])
        if NZ > 0 : 
            DNZ = len(Ui)//NZ + 1
            s = slice(0, -1, DNZ)
            plot_obj[2].set_data(WD_rad[s], MH[s])
    else :
        plot_obj = [None, None, None]
        # Create a hodograph plot
        #ax.set_projection("polar")
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(90)
        #ax.set_yticklabels([])
        if color is None : 
            color = "k"
            for c in ["r", "m", "g", "b", "c", "y"] :
                if c in style :
                    color = c
        plot_obj[0], = ax.plot(WD_rad, MH, style, color=color, label=label, **kwargs_plt)
        plot_obj[1] = ax.plot(WD_rad[0], MH[0], "o", color=color)
    
        if NZ > 0 : 
            DNZ = len(Ui)//NZ + 1
            s = slice(0, len(Ui)+1, DNZ)
            plot_obj[2] = ax.plot(WD_rad[s], MH[s], ".", color=color)
            
    return plot_obj
    
def ax_nighttime(ax, X, where, marg=None, fac=0.04, kwargs_plt_night=default_params["NIGHTTIME"]["kwargs_plt_night"], 
                 kwargs_plt_day=default_params["NIGHTTIME"]["kwargs_plt_day"], **kwargs) :
    """
    Plot a day/night bar below the temporal plots, the bar is yellow during day and dark blue during night
    """
    ymin, ymax = ax.get_ylim()
    if marg is None : 
        marg = ymin-(ymax-ymin)*fac
    if debug : print(ymin, ymax, marg)
    ax.fill_between(X, marg, ymin, where=np.ones(np.array(where).shape), **kwargs_plt_day)
    ax.fill_between(X, marg, ymin, where=where, **kwargs_plt_night)
    ax.set_ylim([marg, ymax])

def ax_landsea(ax, X, where, it=None, marg=None, fac=0.01, kwargs_plt_land=default_params["LANDSEA"]["kwargs_plt_land"], 
                 kwargs_plt_sea=default_params["LANDSEA"]["kwargs_plt_sea"], **kwargs) :
    """
    Plot a land/sea bar below 2DV plots
    """
    if it is not None :
        wherei = where[it]
    else :
        wherei = where
    dx = np.nanmin(np.diff(X))
    ymin, ymax = ax.get_ylim()
    if marg is None : 
        marg = ymin-(ymax-ymin)*fac
    if debug : print(ymin, ymax, marg)
    ax.fill_between(X, marg, ymin, where=np.ones(X.shape), **kwargs_plt_sea)
    ax.fill_between(X, marg, ymin, where=wherei>0.5, **kwargs_plt_land)
    ax.set_ylim([marg, ymax])
    
def ax_WD_hist(ax, X, ylim=None, label=None, it=None, plot_obj=None, kwargs_plt=default_params["WDH"]["kwargs_plt"], **kwargs):
    if it is not None :
        Xi = X[it]
    else :
        Xi = X
    if plot_obj is not None :
        # generated with ChatGPT
        for bar, height in zip(plot_obj.patches, np.histogram(Xi, bins=plot_obj[1])[0]):
            bar.set_height(height)
    else :
        # generated with ChatGPT
        #ax.set_projection("polar")
        plot_obj = ax.hist(np.radians(Xi), label=label, **kwargs_plt)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(90)
        ax.set_yticklabels([])
    return plot_obj
    
def ax_date(ax, x, y, Z, it=None, plot_obj=None, kwargs_plt=default_params["DATE"]["kwargs_plt"], **kwargs):
    if it is not None :
        Zi = Z[it]
    else :
        Zi = Z
        
    if type(Zi) is str :
        pass
    elif type(Zi) in [datetime.datetime, np.datetime64] :
        if "fmt" in kwargs :
            Zi = manage_time.date_to_str(Zi, kwargs["fmt"])
        else :
            Zi = manage_time.date_to_str(Zi)
    elif type(Zi) in [datetime.timedelta, np.timedelta64] :
        if "fmt" in kwargs :
            Zi = manage_time.timedelta_to_str(Zi, kwargs["fmt"])
        else :
            Zi = manage_time.timedelta_to_str(Zi)
    else :
        Zi = str(Zi)
        
    if type(x) is str :
        x0, x1 = ax.get_xlim()
        dx = x1-x0
        if x == "right" :
            x = x1-0.2*dx
        else :
            x = x0+0.01*dx
    if type(y) is str :
        y0, y1 = ax.get_ylim()
        dy = y1-y0
        if y == "bottom" :
            y = y0+0.05*dy
        else :
            y = y1-0.05*dy
            
    if plot_obj is not None :
        plot_obj.set_text(Zi)
    else :
        plot_obj = ax.text(x, y, Zi, **kwargs_plt)
    return plot_obj

def ax_vline(ax, X, label=None, it=None, plot_obj=None, kwargs_plt=default_params["AXVLINE"]["kwargs_plt"], **kwargs):
    return ax.axvline(x=X, label=label, **kwargs_plt)

def ax_hline(ax, Y, label=None, it=None, plot_obj=None, kwargs_plt=default_params["AXHLINE"]["kwargs_plt"], **kwargs):
    return ax.axhline(y=Y, label=label, **kwargs_plt)

def ax_vspan(ax, X1, X2, label=None, it=None, plot_obj=None, kwargs_plt=default_params["AXVSPAN"]["kwargs_plt"], **kwargs):
    return ax.axvspan(X1, X2, label=label, **kwargs_plt)

def ax_hspan(ax, Y1, Y2, label=None, it=None, plot_obj=None, kwargs_plt=default_params["AXHSPAN"]["kwargs_plt"], **kwargs):
    return ax.axhspan(Y1, Y2, label=label, **kwargs_plt)

def get_cmap_extend(clim, Z, cmap, discrete=None, ticks=None, nancolor=None):
    if clim is None :
        vmin = np.nanmin(Z)
        vmax = np.nanmax(Z)
        extend = "neither"
    elif type(clim) is list:
        vmin, vmax = clim
        if vmin > np.nanmin(Z) :
            if vmax < np.nanmax(Z) :
                extend = "both"
            else :
                extend = "min"
        elif vmax < np.nanmax(Z) : 
            extend = "max"
        else :
            extend = "neither"
    elif clim == "centered" :
        vmax = max(np.nanmax(Z), -np.nanmin(Z))
        vmin = -vmax
        extend = "neither"
    if discrete is not None : 
        n = discrete
        Zmin = np.nanmin(Z)
        Zmax = np.nanmax(Z)
        dZ = (vmax - vmin)/n
        if not np.isnan(vmin):
            if ticks is None or ticks == "out" :
                ticks = np.arange(vmin, vmax+1e-10, dZ)
            elif ticks == "in" :
                ticks = np.arange(vmin, vmax+1e-10, dZ)
                vmin -= dZ/2
                vmax += dZ/2
                n += 1
            if Zmin < vmin-1e-10 :
                ticks = np.delete(ticks, 0)
            if Zmax > vmax+1e-10:
                ticks = np.delete(ticks, -1)
        cmap = get_cmap(cmap, n)
    else : 
        cmap = get_cmap(cmap)
    if nancolor is not None :
        cmap.set_bad(color=nancolor)
    return vmin, vmax, extend, cmap, ticks

def get_cmap(i=0, n=None) :
    if type(i) is int :
        if i in tab_cmap :
            return get_cmap(tab_cmap[i], n) 
        else : 
            return get_cmap(tab_cmap[0], n)
    # Useless : juste type _r at the end of the colormap
    elif type(i) is str :
        if i.startswith("inversed_") :
            return inversed_cmap(i[9:], N=n)
    return plt.get_cmap(i, n)

def color_from_cmap(cmap, i, n):
    cmap = get_cmap(cmap, n)
    return cmap(i/(n-1))

def nice_number(value, round_=False):
    """
    nice_number and nice_bounds come from :
    https://stackoverflow.com/questions/4947682/intelligently-calculating-chart-tick-positions
    """
    '''nice_number(value, round_=False) -> float'''
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5:
            nice_fraction = 1.
        elif fraction < 3.:
            nice_fraction = 2.
        elif fraction < 7.:
            nice_fraction = 5.
        else:
            nice_fraction = 10.
    else:
        if fraction <= 1:
            nice_fraction = 1.
        elif fraction <= 2:
            nice_fraction = 2.
        elif fraction <= 5:
            nice_fraction = 5.
        else:
            nice_fraction = 10.

    return nice_fraction * 10 ** exponent


def nice_bounds(axis_start, axis_end, num_ticks=10):
    '''
    nice_bounds(axis_start, axis_end, num_ticks=10) -> tuple
    @return: tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    '''
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick = nice_number(nice_range / (num_ticks - 1), round_=True)
        axis_start = math.floor(axis_start / nice_tick) * nice_tick
        axis_end = math.ceil(axis_end / nice_tick) * nice_tick

    return axis_start, axis_end, nice_tick

def test_colormaps():
    params = []
    for v in plt.colormaps() :
        if v[-2:] not in ["_r"] :
            params.append({
                "X" : np.arange(50), "Y" : np.arange(50), "Z" : np.arange(50), "kwargs_plt" : {"cmap" : v}, "discrete" : 6, "title" : v, "DY_subplots" : 3, "NX_subplots" : 6,
                "yticks" : [], "xticks" : [], "typ" : "SCATTER",
            })
    fig = plot_fig(params)