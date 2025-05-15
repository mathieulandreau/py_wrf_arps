from ..class_proj import Proj
from ..WRF_ARPS import Dom
from ..lib import manage_list

def save_PSL(dom, nprocs=1):
    """write_postproc of PSL
    Parameters
        dom(Dom) : domain object (see class_dom)
    Optional
        nprocs(int) : 1 = serial, >1 = multiprocessing
    14/05/2025 : Mathieu LANDREAU
    """
    varname = "PSL"
    NT = dom.get_data("NT_HIST")
    print(NT)
    def temp(dom, it) :
        print(it, end=" ")
        var = dom.get_data(varname, itime=it)
        dom.write_postproc(varname, var, ('y', 'x'), itime=it, long_name="Sea-level pressure", standard_name=varname, units="Pa", latex_units="Pa", typ=np.float32)
    if nprocs > 1 :
        from multiprocessing import Pool, cpu_count
        inputs = [(dom, it) for it in range(NT)]
        with Pool(processes=nprocs) as pool:
            pool.starmap(temp, inputs)
    else :
        for it in range(NT) :
            temp(dom, it)
            
def save_PSL24(dom, nprocs=1):
    """write_postproc of PSL
    Parameters
        dom(Dom) : domain object (see class_dom)
    Optional
        nprocs(int) : 1 = serial, >1 = multiprocessing
    14/05/2025 : Mathieu LANDREAU
    """
    varname = "PSL24"
    NT = dom.get_data("NT_HIST")
    PSL = dom.get_data("PSL", itime="ALL_TIMES")
    PSL24 = manage_list.moving_average2(PSL, 24*6)
    print(PSL24.shape)
    print(NT)
    def temp(dom, it, PSL24_it) :
        print(it, end=" ")
        dom.write_postproc(varname, PSL24_it, ('y', 'x'), itime=it, long_name="day-avg sea-level pressure", standard_name=varname, units="Pa", latex_units="Pa", typ=np.float32)
    if nprocs > 1 :
        inputs = [(dom, it, PSL24[it]) for it in range(NT)]
        with Pool(processes=nprocs) as pool:
            pool.starmap(temp, inputs)
    else :
        for it in range(NT) :
            temp(dom, it, PSL24[it])
