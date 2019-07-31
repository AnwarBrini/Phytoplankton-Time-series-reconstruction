# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:51:34 2019

@author: abrini
"""
import numpy as np
import netCDF4 as nc
import os

def read_table(path, variable, year, month):
    """
    Read Variable

    Parameters
    ----------
    path : string
    variable : string
        variable name.
    year : int
    month : int

    Returns
    -------
    dict
        variable name: ndarray of the variable.

    """
    var_file_path = path + "/" + str(year) + "/" + str(month) + "/" + str(variable) +"_" + str(year) +"{:02d}".format(month)+".nc"
    var_file = nc.Dataset(var_file_path)
    var = var_file.variables[str(variable)][:].data
    return {variable:var} 

def loop_lon_lat(lat_len, lon_len):
    """
    Loop over latitudes and longitudes in one loop

    Parameters
    ----------
    lat_len : int
    lon_len : int

    Yields
    ------
    i : int
        lat
    j : int
        lon

    """
    for i in range(lat_len):
        for j in range(lon_len):
            yield i, j

def mon_sin(x):
    """
    Parameters
    ----------
    x : numpy ndarray
        months.

    Returns
    -------
    TYPE numpy ndarray
    """
    return np.sin(2*np.pi*x/12)

def mon_cos(x):
    """
    Parameters
    ----------
    x : numpy ndarray
        months.

    Returns
    -------
    TYPE numpy ndarray
    """
    return np.cos(2*np.pi*x/12)

def recreate_map(model, gendata_path, mean_, scale_, feature_names=None,
                 start=2012, end=2016, shape=[1,13], lat_sin=False, verbose=False):
    """
    Creates numpytxt files of chl predictions in gendata_path using model ignoring rows with missing values
    model - the model used to predict
    feature_names - list of features to be used in the predictions
    start - start year of the predictions
    end - end year of the predictions
    mean_ - the mean used in the standard scaling to do the backtransformation
    scale_ - the std used in the standard scaling to do the backtransformation
    """
    path = "./data/INPUT_AI/"
    lon_lat_path = "./data/INPUT_AI/1999/10/chl_199910.nc"
    ncfile = nc.Dataset(lon_lat_path)
    lons = ncfile.variables['lon'][:].data
    lats = ncfile.variables['lat'][:].data
    ncfile.close()
    chls = []
    if feature_names is None:
        feature_names = ["sla", "sst", "uera", "vera", "u", "v", "sw"]
    try:
        os.mkdir("./" + gendata_path + "{:d}-{:d}".format(start,end))
        print("Directory " , "./" + gendata_path + "{:d}-{:d}".format(start,end)
        ,  " Created ")
    except FileExistsError:
        print("Directory " , "./"+gendata_path+"{:d}-{:d}".format(start,end),
              " already exists")
            
    for year in range(start, end):
        months = [int(item) for item in os.listdir(path+"/"+str(year)) ]
        months.sort()
        for month in months:
            data = [read_table(path, item, year, month) for item in feature_names ]
            chl = np.full([178 ,358], -9999.0)
            for i, j in loop_lon_lat(178, 358):
                features = [data[index][feat_name][0, i, j] for index, feat_name in enumerate(feature_names)]
                if -9999.0 in features:
                    continue
                if lat_sin:
                    lat = np.sin(lats[i])
                else:
                    lat = lats[i]
                lon1 = np.sin(lons[j])
                lon2 = np.cos(lons[j])
                mon_1 = mon_sin(int(month))
                mon_2 = mon_cos(int(month))
                #lat, year, sla , sst, uera, vera , u, v, sw,lon1,lon2,mon_1,mon_2
                features.insert(0,lat)
                features.insert(1,year)
                features.insert(9,lon1)
                features.insert(10,lon2)
                features.insert(11,mon_1)
                features.insert(12,mon_2)
                features = ( np.array(features,)- mean_ ) / scale_
                chl[i,j] = model.predict(np.reshape(features,shape))
            if verbose:
                print(str(year)+" "+str(month))
            chls.append(chl)
            np.savetxt(gendata_path + "{:d}-{:d}".format(start, end) +
                       "/Chl_" + str(year) + "{:02d}".format(month) + ".dat", chl)
    return np.stack(chls, axis=0)



