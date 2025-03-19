# py\_wrf\_arps

This project contains python class and scripts to extract and plot WRF and ARPS output data. It has been used to process WRF results, AROME data, and LiDAR data for the following paper:
Landreau, Mathieu, Boris Conan, Benjamin Luce, et Isabelle Calmet. 2025. « Observation and LES modeling of an offshore atmospheric undular bore generated during sea-breeze initiation near a peninsula ». https://hal.science/hal-04996636.

## How to Use 

* Create a conda environment
```
    conda env create -f env/py_wrf_arps.yml
```
* Add an environment variable $PYTHONPATH that contains the name of this folder in your bashrc
```
    PYTHONPATH=Path/to/this/folder/:$PYTHONPATH
```
* Import py\_wrf\_arps in your python script
```
    import py_wrf_arps
```
* Modify the different path in files lib/manage\_path and classProj to link to your simulations and data files

## Example

Few Jupyter notebooks (related to the paper) using this library can be found in [https://github.com/mathieulandreau/Landreau_Conan_Luce_Calmet_a1](https://github.com/mathieulandreau/Landreau_Conan_Luce_Calmet_a1)


