import git
import datetime
print("You are using py_wrf_arps, the current git hash is : ", git.Repo(__file__, search_parent_directories=True).head.object.hexsha, ", the current date is : ", datetime.datetime.now())
from .lib import *
from .post import *
from .class_variables import *
from .WRF_ARPS import *
from .expe_data import *
from .class_proj import Proj
