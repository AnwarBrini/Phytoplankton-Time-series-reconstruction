# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:26:41 2019

@author: abrini
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from mpl_toolkits.basemap import Basemap,cm
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import os
import glob




def rmse(targets,predictions):
    return np.sqrt(np.nanmean((predictions-targets)**2))

def nrmse(targets,predictions):
    return np.sqrt(np.nanmean((predictions-targets)**2))/(np.nanmean(targets))

def rmse_chl(targets,predictions):
    """
    returns Root Mean Squared Error of predicted chl and observed chl on the global scale
    return format - numpy array of shape (Latitudes,longitutdes)
    targets -  numpy array of Observations of shape (timesteps, lat,lon)
    predictions - numpy array of predicted values (timesteps, lat,lon)
    """
    n_t, n_x, n_y = predictions.shape
    mse = np.zeros((n_x,n_y))
    for i in range(0, n_x):
        for j in range(0, n_y):
            mse[i,j] = rmse(predictions[:,i,j], targets[:,i,j])
    
    return mse

def nrmse_chl(targets,predictions):
    """
    returns Normalized Root Mean Squared Error of predicted chl and observed chl on the global scale
    return format - numpy array of shape (Latitudes,longitutdes)
    targets -  numpy array of Observations of shape (timesteps, lat,lon)
    predictions - numpy array of predicted values (timesteps, lat,lon)
    
    Using nrmse function to compute the NRMSE per pixel
    """
    n_t, n_x, n_y = predictions.shape
    mse = np.zeros((n_x,n_y))
    for i in range(0, n_x):
        for j in range(0, n_y):
            mse[i,j] = nrmse(predictions[:,i,j], targets[:,i,j])
    
    return mse

def cross_correlation(vars1, vars2):
    """
    returns Pearson crosscorrelation between two variables
    return format - numpy array of shape (lat,lon)
    vars1 -  numpy array of var1 of shape (timesteps, lat,lon)
    vars2 - numpy array of var2 of shape(timesteps, lat,lon)
    """
    n_t, n_x, n_y = vars1.shape
    corr = np.zeros((n_x,n_y))
    for i in range(0, n_x):
        for j in range(0, n_y):
            corr[i,j] = ma.corrcoef(ma.masked_invalid(vars1[:,i,j]), ma.masked_invalid(vars2[:,i,j]))[0, 1]
    
    return corr

def get_gen_chl(folder):
    """
    reads numpytxt files of generated chl concentrations
    folder - path to the numpytxt files
    """
    files = os.listdir("./"+folder);files.sort()
    chls = []
    for file in files:
        chls.append(np.loadtxt("./"+folder+"/"+file))
    
    return np.stack(chls,axis=0)

def get_mask(variable):
    vares = []
    path = "./data/INPUT_AI/"
    for year in os.listdir(path):
        for month in os.listdir(path+"/"+year):
            var_file_path = path + "/" + year + "/" + month + "/" + str(variable) +"_" + year +"{:02d}".format(int(month))+".nc"
            var_file = nc.Dataset(var_file_path)
            var = var_file.variables[str(variable)][:].data
            var[var != -9999.0] = 0
            var[var == -9999.0] = 1
           # #chl = chl.astype(np.float64)
            #ms = mean_squared_error((chl), np.expand_dims(ms,axis=0) )
            vares.append(var[0])
    return np.stack(vares,axis=0)

def get_var(variable,path="./data/INPUT_AI/",start=1998,end=2016):
    vares = []
    for year in range(start,end):
        months = os.listdir(path+"/"+str(year))
        months.sort(key=lambda x:"{:02d}".format(int(x)) )
        for month in months:
            var_file_path = path + "/" + str(year) + "/" + month + "/" + str(variable) +"_" + str(year) +"{:02d}".format(int(month))+".nc"
            var_file = nc.Dataset(var_file_path)
            var = var_file.variables[str(variable)][:].data
            var[var == -9999.0] = np.NaN
           # #chl = chl.astype(np.float64)
            #ms = mean_squared_error((chl), np.expand_dims(ms,axis=0) )
            vares.append(var[0])
    return np.stack(vares,axis=0)

def sort_over_folders(path):
    files = glob.glob(path+"/*/*/chl*")
    n_files = []
    r_files = []
    for file in files:
        s = file.split("/")
        s[3] = "{:02d}".format(int(s[3]))
        s = '/'.join(s)
        n_files.append(s)
    n_files.sort()
    for file in n_files:
        s = file.split("/")
        s[3] = "{:01d}".format(int(s[3]))
        s = '/'.join(s)
        r_files.append(s)
    return r_files 

def heat_map(data):
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(32, 9))
    sns.heatmap(corrmat, cmap='viridis');

def print_map(var_corr,var_name,sub=111,vmin=None,vmax=None,save=False,lon_lat_path="./data/INPUT_AI/1999/10/chl_199910.nc"):
    """
    returns plot object of data map with mercator projection and centrelized pacific
    var_corr - variable to plot on the map of shape (178,358)
    var_name - title to display
    vmin - minimum value to show on the scale
    vmax - maximum value to show on the scale
    sub - subplot value
    """
    plt.subplot(sub)

    map = Basemap(projection='merc',llcrnrlon=20,urcrnrlon=380,llcrnrlat=-60,urcrnrlat=60,resolution="l")
    label_name = var_name

    path = lon_lat_path
    ncfile = nc.Dataset(path)
    lons = ncfile.variables['lon'][:].data
    lats = ncfile.variables['lat'][:].data
    xx, yy = np.meshgrid(lons, lats)

    if vmin == None:
        vmin = np.min(var_corr.squeeze())
    if vmax == None:
        vmax = np.max(var_corr.squeeze())
        
    cmap = plt.get_cmap("summer")
    map.drawcoastlines()
    map.fillcontinents('#cccccc')
    colormesh = map.pcolormesh(xx, yy, var_corr,latlon=True, vmin = vmin, vmax =vmax, cmap=cm.sstanom)
   #contour = map.contour(xx, yy,(var_corr),  np.arange(-1, 1, 0.3 ),latlon=True, linestyles = 'solid')
    map.colorbar(colormesh)
    cb = map.colorbar(colormesh, location='bottom', label=label_name)
    map.drawmeridians(np.arange(1.5,358.5,100.0),labels=[0,1,1,0]) #longitudes
    map.drawparallels(np.arange(-89.0,88.0,30.0),labels=[1,0,0,1]) #latitudes
    #cb.add_lines(contour)
    #cb.set_ticks([-10, -5, 5, 30])
    if save == True:
        plt.savefig("./figs/"+label_name+'.png')
        
        
def print_missingmap(var,var_name,sub=111,save=False):
    """
    returns plot object of data map with mercator projedef get_var(variable,start=1998,end=2016):
    vares = []
    path = "./data/INPUT_AI/"
    for year in range(start,end):
        for month in os.listdir(path+"/"+str(year)):
            var_file_path = path + "/" + str(year) + "/" + month + "/" + str(variable) +"_" + str(year) +"{:02d}".format(int(month))+".nc"
            var_file = nc.Dataset(var_file_path)
            var = var_file.variables[str(variable)][:].data
            var[var == -9999.0] = np.NaN
           # #chl = chl.astype(np.float64)
            #ms = mean_squared_error((chl), np.expand_dims(ms,axis=0) )
            vares.append(var[0])
    return np.stack(vares,axis=0)ction and centrelized pacfic
    of missing value
    var - variable to plot on the map of shape (178,358)
    var_name - title to display
    vmin - minimum value to show on the scale
    vmax - maximum value to show on the scale
    sub - subplot value
    save - bool to save figure
    """    
    lon_lat_path="./data/INPUT_AI/1999/10/chl_199910.nc"
    path = lon_lat_path
    ncfile = nc.Dataset(path)
    lons = ncfile.variables['lon'][:].data
    lats = ncfile.variables['lat'][:].data
    xx, yy = np.meshgrid(lons, lats)
    plt.subplot(sub)
    map = Basemap(projection='merc',llcrnrlon=20,urcrnrlon=380,llcrnrlat=-80,urcrnrlat=80,resolution="l")
    label_name = var_name+ " Missing values"

    cmap = plt.get_cmap("summer")
    map.drawcoastlines()
    map.fillcontinents('#cccccc')
    colormesh = map.pcolormesh(xx, yy, var,latlon=True, vmin = np.min(var.squeeze()), vmax =np.max(var.squeeze()), cmap=cm.sstanom)
    contour = map.contour(xx, yy,(var),  np.arange(-1, 1, 0.3 ),latlon=True, linestyles = 'solid')
    map.colorbar(colormesh)
    cb = map.colorbar(colormesh, location='bottom', label=label_name)
    cb.add_lines(contour)
    map.drawmeridians(np.arange(1.5,358.5,100.0),labels=[0,1,1,0]) #longitudes
    map.drawparallels(np.arange(-89.0,88.0,30.0),labels=[1,0,0,1]) #latitudes
    #cb.set_ticks([-10, -5, 5, 30])
    #plt.savefig("./figs/"+label_name+'.png')
    
    if save == True:
        plt.savefig("./figs/"+label_name+'.png')
    
        
    
