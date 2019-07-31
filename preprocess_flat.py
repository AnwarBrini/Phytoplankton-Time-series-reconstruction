# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:51:17 2019

@author: abrini
"""

import numpy as np
import pandas as pd
import netCDF4 as nc

# Adjusting Coords lat(0=-80;177=88) and lon(357=358.5;0=1.5) 

def lat_transform(x):
    return x - 89

def lon_transform(x):
    return x + 1.5

def mon_sin(x):
    return np.sin(2*np.pi*x/12)

def mon_cos(x):
    return np.cos(2*np.pi*x/12)

def lon_lat_retrieve(lat, lon_1, lon2):
    llat = np.arcsin(lat)
    if (lon_1 > 0 ):
        lon = np.arccos(lon2) + np.pi
    else:
        lon = np.arccos(lon2)
    return llat, lon


def preprocess_flat_data(data):
    
    try:
        data = data.drop(columns=["Unnamed: 0"])
    except:
        pass
    
    
    data = data[data["hl_"] != -9999.0]
    data = data[data["sla_"] != -9999.0]
    data = data[data["u_"] != -9999.0]
    data = data[data["v_"] != -9999.0]
    data = data[data["uera_"] != -9999.0]
    data = data[data["vera_"] != -9999.0]

    #data[data == -9999.0] = 0
 #   data["lon"] = data["lon"].apply(lon_transform)
#   data["lat"] = data["lat"].apply(lat_transform)
    data["lat"] = data["lat"].apply(np.sin)
    data["lon_1"] = data["lon"].apply(np.sin)
    data["lon_2"] = data["lon"].apply(np.cos)
    data["mon_1"] = data["mon"].apply(mon_sin)
    data["mon_2"] = data["mon"].apply(mon_cos)
    #data["hl_"] = data["hl_"].apply(np.log)
    data = data.drop(columns=["lon","mon"])
    return data



def preprocess_pix(data):

    
    data = data[data["hl_"] != -9999.0]
    data = data[data["sla_"] != -9999.0]
    data = data[data["u_"] != -9999.0]
    data = data[data["v_"] != -9999.0]
    data = data[data["uera_"] != -9999.0]
    data = data[data["vera_"] != -9999.0]

    #data[data == -9999.0] = 0
 #   data["lon"] = data["lon"].apply(lon_transform)
#   data["lat"] = data["lat"].apply(lat_transform)
    data["lat"] = data["lat"].apply(np.sin)
    data["lon_1"] = data["lon"].apply(np.sin)
    data["lon_2"] = data["lon"].apply(np.cos)
    data["mon_1"] = data["mon"].apply(mon_sin)
    data["mon_2"] = data["mon"].apply(mon_cos)
    #data["hl_"] = data["hl_"].apply(np.log)
    data = data.drop(columns=["lon","mon"])
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    l = list(data.columns)
    l.pop(0)
    Y = np.log(data["hl_"])
    X = data[l]
    s = StandardScaler()
    scaled_X = s.fit_transform(X.values)
    X_train, X_val, y_train, y_val = train_test_split(scaled_X, Y, test_size=0.2, random_state=0)
    return X_train, X_val, y_train, y_val, s
    
    
