# py\_wrf\_arps

This project contains python class and scripts to extract and plot WRF and ARPS output data. It has been used to process WRF results, AROME data, and LiDAR data for the following paper:
Landreau, M., Conan, B., Luce, B., Calmet, I., 2025, Observation and Large-Eddy Simulation of an Offshore Atmospheric Undular Bore During Sea-Breeze Initiation
[https://doi.org/10.1029/2025JD043852](https://doi.org/10.1029/2025JD043852)

## Install
* Create a conda (or uv) environment
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

## Warning

This project has been built during the PhD of Mathieu Landreau (2022-2025) and is far from perfect. Notably, the DomARPS object has not been used for a while. The user should always question the results, and the code might be hard to understand for anyone else. A few explanations are given hereafter 

## Description

The py\_wrf\_arps projects aims at processing data from WRF simulations, ARPS simulations, AROME, LiDAR, Meteo France stations, and other datasets, and plotting figures. The main idea is to homogenize the data (use the same variable names, the same coordinate system, the same date format), and wrap the lecture of the data into a single function **get_data**. There are two types of data objects: the **Dom**, managing the gridded data (e.g. results on simulation domains), and the **Expe**, managing the experimental data (more flexible). These classes use a method **get_data** to read the variables from output files and to post-process them. An object **Proj** contains several Dom or Expe objects and is used to draw the Figures, to perform comparison between the different dataset. The **Variable** objects contains information about the variables from the different datasets (name, units) and functions to read the files. The **lib** folder contains several pythons methods that are used in several occasions in the classes to manage the time, the projections, ... Finally, the **post** folder contains some post-processing methods and classes, built using the py\_wrf\_arps library, that are more specifically related to the PhD results.

### Dom

The **Dom** object has some inherited classes for each code: **DomWRF** for WRF results, **DomARPS** for ARPS, and **DomAROME** for AROME. In addition, the wrfinput files can be read with the **DomWRFinput** object inherited from DomWRF. Note that DomARPS is abandoned as no ARPS simulation has been performed for a while. In theory, Dom should contain all the methods that are common for every inherited class.


## Example

Few Jupyter notebooks (related to the paper) using this library can be found in [https://github.com/mathieulandreau/Landreau_Conan_Luce_Calmet_a1](https://github.com/mathieulandreau/Landreau_Conan_Luce_Calmet_a1)

## Possible improvements
A non-exhaustive list of possible improvements : 
- The initiation of the Proj object should be more flexible (especially the paths).
- The computation of derivatives is very slow and could be improved (and probably other computations as well).
- The Expe object could be a Data object from which the Dom object could inherit.
- The limit between Dom and DomWRF objects is probably unclear because I mostly analyzed WRF results.
- The functions calculate should include some user-defined "modes" because the output data of a code (e.g. WRF) can change depending on the simulation settings.
- All the data are converted to numpy array where the "nature" of the dimensions (t, x, y, or z) is lost so that there is sometimes a need to "guess" the dimension. Using xarray could make the code more robust. However, I don't know if all the calculations of py\_wrf\_arps are possible with xarray
