# py\_wrf\_arps

This project contains python class and scripts to extract and plot WRF and ARPS output data

## How to Use 

* Add an environment variable $PYTHONPATH that contains the name of this folder in your bashrc
```
    PYTHONPATH=Path/to/this/folder/:$PYTHONPATH
```
* Import py\_wrf\_arps in you python script
```
    import py_wrf_arps
```
* Create a conda environment
```
    conda env create -f env/py_wrf_arps.yml
```
* Modify the different path in files lib/manage\_path and classProj to link to your simulations and data files


