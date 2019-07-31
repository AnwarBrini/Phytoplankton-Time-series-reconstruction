#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:50:58 2019

@author: abrini
"""

import argparse
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc

def get_paths(path):
    """
    Get sorted file paths using glob

    Parameters
    ----------
    path : STRING
        Regex format for each corresponding variable.

    Returns
    -------
    LIST
        list of all files corresponding to the regex format sorted by filename.

    """
    return sorted(glob.glob(path), key=lambda x: x.split('/')[-1])

def read_flat_file(var_path, var_name, f_type):
    """
    

    Parameters
    ----------
    var_path : TYPE
        DESCRIPTION.
    var_name : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if f_type:
        var_data = nc.Dataset(var_path).variables[var_name][:].data
    else:
        var_data = np.load(var_path)
    _, nlat, nlon = var_data.shape
    return np.reshape(var_data, nlat*nlon)

def read_all_paths(input_path):
    """
    Read All files paths of the different variables
    
    Parameters
    ----------
    input_path : String
        path of folder containing all data in a regex format.
    Returns
    -------
    list
        list of list of paths.

    """
    chl_paths = get_paths("{}/chl_*[0-9].*".format(input_path))
    sla_paths = get_paths("{}/sla_*[0-9].*".format(input_path))
    sst_paths = get_paths("{}/sst_*[0-9].*".format(input_path))
    sw_paths = get_paths("{}/sw_*[0-9].*".format(input_path))
    u_paths = get_paths("{}/u_*[0-9].*".format(input_path))
    v_paths = get_paths("{}/v_*[0-9].*".format(input_path))
    uera_paths = get_paths("{}/uera_*[0-9].*".format(input_path))
    vera_paths = get_paths("{}/vera_*[0-9].*".format(input_path))
    paths = [chl_paths, sla_paths, sst_paths, sw_paths, u_paths,
            v_paths, uera_paths, vera_paths]
    return paths

def transform_nc(input_path, parmcols, start_y, end_y, f_type):
    """
    Read nc files and transform them to pandas dataframe

    Parameters
    ----------
    input_path : String
        path of folder containing all data in a regex format.
    parm_cols : list
        list of different parms to load from the folder.

    Returns
    -------
    Pandas dataframe containing flattened data.

    """
    print("Loading paths")
    paths = read_all_paths(input_path)
    t_steps = (end_y - start_y + 1) * 12
    lats_data = np.repeat(nc.Dataset(paths[0][0]).variables['lat'][:], 358).data
    lons_data = np.repeat(nc.Dataset(paths[0][0]).variables['lon'][:], 178).data
    lon_1_data = np.sin(lons_data)
    lon_2_data = np.cos(lons_data)
    data = []
    print("Loading Data for each item step")
    for t in range(t_steps):
        year_data = (t // 12) + start_y * np.ones(lats_data.shape[0])
        mon_1_data = np.sin(2*np.pi*(t%12)/12) * np.ones(lats_data.shape[0])
        mon_2_data = np.cos(2*np.pi*(t%12)/12) * np.ones(lats_data.shape[0])
        chl_data = read_flat_file(paths[0][t], 'chl', f_type)
        sla_data = read_flat_file(paths[1][t], 'sla', f_type)
        sst_data = read_flat_file(paths[2][t], 'sst', f_type)
        sw_data = read_flat_file(paths[3][t], 'sw', f_type)
        u_data = read_flat_file(paths[4][t], 'u', f_type)
        v_data = read_flat_file(paths[5][t], 'v', f_type)
        uera_data = read_flat_file(paths[6][t], 'uera', f_type)
        vera_data = read_flat_file(paths[7][t], 'vera', f_type)
        t_data = [chl_data, lats_data, year_data, sla_data, sst_data, uera_data,
                  vera_data, u_data, v_data, sw_data, lon_1_data, lon_2_data,
                  mon_1_data, mon_2_data]
        data.append(pd.DataFrame(np.column_stack(t_data), columns=parmcols))
    print("Concatenation of all time steps")
    data = pd.concat(data, ignore_index=True)
    return data

def get_args():
    """
    Args

    -------

    """
    parser = argparse.ArgumentParser(description='Transforming 3D data to 2D data')
    parser.add_argument('-i', '--input', help='folder containg the data')
    parser.add_argument('-o', '--output', help='path for output csv file')
    parser.add_argument('--start-year', default=1998, help='starting year of the data')
    parser.add_argument('--end-year', default=2015, help='ending year of the data')
    parser.add_argument('--ftype', default=0, help='0 if netcdf files; 1 if npy files')
    args = parser.parse_args()
    return  args

def main():
    """
    Main func
    -------

    """
    args = get_args()
    # the trained models uses the following order of variables as input
    # lat, year, sla , sst, uera, vera , u, v, sw, lon1, lon2, mon_1, mon_2
    parmcol = ["chl", "lat", "year", "sla", "sst", "uera", "vera", "u",
               "v", "sw", "lon_1", "lon_2", "mon_1", "mon_2"]
    start_y = args.start_year
    end_y = args.end_year
    f_type = args.ftype
    out_data = transform_nc(args.input, parmcol, start_y, end_y, f_type)
    print("Saving file {}".format(args.output))
    out_data.to_csv(args.output)

if __name__ == "__main__":
    main()