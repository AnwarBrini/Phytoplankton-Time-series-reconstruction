#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:33:22 2019

@author: abrini
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import numpy.ma as ma
#from scipy import signal
from mpl_toolkits.basemap import Basemap


def get_wgts(lats):
    """
    Get lattitude weights from latitudes using mercator projection
    Parameters
    ----------
    lats : numpy.array
        array containing lats.

    Returns
    -------
    wgts : array containing correspondings weights
    """
    wgts   = np.sqrt(np.cos(np.deg2rad(lats)))
    wgts   = wgts.reshape(len(wgts), 1)
    return wgts

def detrend(sig, nt, nlat, nlon):
    """
    Removes missing points and applys linear detrending on the signal
    Parameters
    ----------
    sig : ndarry containing the signal
    nt : int
        time dimension of the signal.
    nlat : int
        latitude dimension of the signal.
    nlon : int
        longitude dimension of the signal.
    Returns
    -------
    sig_detrend : ndarray
        of the detrended signal.

    """
    sig_detrend = np.empty(shape=(nt, nlat, nlon))
    for i in range(0, nlat):
        for j in range(0, nlon):
            sig1 = sig[:, i, j]
            b = ~np.isnan(sig1)
            sig1 = sig1[b]
            if sig1.size == 0:
                continue
            #sig_det = signal.detrend(sig1, axis=0, type='linear', bp=0)
            sig_detrend[b, i, j] = sig1
    return sig_detrend

def seasonal_sig(sig, nt, ny, ndy, nlat, nlon, wgts):
    """
    Extracts Climatology/Seasonal signal 

    Parameters
    ----------
    sig : numpy ndarray
        Signal.
    nt : int
        time dimension.
    ny : int
        number of time elements per season.
    ndy : int
        number of time elements per season..
    nlat : int
        latitude dimension of the signal.
    nlon : int
        longitude dimension of the signal.
    wgts : numpy array
        latitude weights.

    Returns
    -------
    sig1 : numpy ndarray
        seasonal signal.

    """
    season = ma.mean(sig, axis=0)
    print(season.shape)
    dev = ma.std(sig, axis=0)
    seas_signal = sig - season
    seas_signal = seas_signal / dev
    return seas_signal

def interannual_sig(sig, nt, ny, ndy, nlat, nlon, wgts, return_season=False):
    """
    Remove Climatology/Seasonality 

    Parameters
    ----------
    sig : numpy ndarray
        Signal.
    nt : int
        time dimension.
    ny : int
        number of time elements per season.
    ndy : int
        number of time elements per season..
    nlat : int
        latitude dimension of the signal.
    nlon : int
        longitude dimension of the signal.
    wgts : numpy array
        latitude weights.
    return_season : bool, optional
        return climatology. The default is False.

    Returns
    -------
    TYPE numpy ndarray
        interannual signal.

    """
    sig1 = np.reshape(sig, [ny, ndy, nlat, nlon])
    season = np.nanmean(sig1, axis=(0))
    print(season.shape)
    dev = np.nanstd(sig1, axis=0)
    dev[dev == 0 ] = 1
    for i in range(ndy):
        sig1[:, i, :, :] = (sig1[: , i, :, :] - season[i,...]) / dev[i,...]
    sig1 = np.reshape(sig1, [nt, nlat, nlon])
    if return_season:
        return sig1, season, dev 
    else:
        return sig1

def eof(sig, lat, lon, label="", X1=None, scale=True, save=False, neof=3, retrun_eof=False):
    """
    Computes Empirical Orthogonal Function decomposition.
    
    Parameters
    ----------
    sig : numpy ndarray
    nlat : int
        latitude dimension of the signal.
    nlon : int
        longitude dimension of the signal.
    label : strings, optional
        DESCRIPTION. The default is "".
    X1 : numpy ndarray, optional
        second signal to project on the first's signal decomposition base. The default is None.
    scale : TYPE, bool
        Normalize signal using standardScaler. The default is True.
    save : bool, optional
        save eofs figures. The default is False.
    neof : int, optional
        number of eof modes to plot. The default is 3.
    retrun_eof : bool, optional
        return eof values. The default is False.

    Returns
    -------
    TYPE ndarray
        eof values.

    """
    plt.ioff()
    nt, nlat, nlon = sig.shape
    ny = 2016-1998
    startY = 1998
    endY = 2016
    X = np.reshape(sig, (sig.shape[0], len(lat) * len(lon)), order='F')
    print(np.any(np.isnan(X)))
    X = ma.masked_array(X, np.isnan(X))
    land = X.sum(0).mask
    ocean = ~land
    X = X[:,ocean]  
    if scale:
        print("mean Before Scaling = {:f}".format(X.mean()))
        print("std Before Scaling = {:f}".format(X.std()))
        from sklearn import preprocessing
        scaler  = preprocessing.StandardScaler()
        scaler_chl = scaler.fit(X)
        X = scaler_chl.transform(X)
        print("mean After Scaling = {:f}".format(X.mean()))
        print("std After Scaling = {:f}".format(X.std()))
    from sklearn.decomposition import pca
    skpca = pca.PCA()
    skpca.fit(X)
    ipc = np.where(skpca.explained_variance_ratio_.cumsum() >= 0.70)[0][0]
    print("{:d} modes represent 70% of the signal".format(ipc))
    if X1 is None:
        PCs = skpca.transform(X)
        PCs = PCs[:, :ipc]
    else:
        X1 = np.reshape(X1, (X1.shape[0], len(lat) * len(lon)), order='F')
        print(np.any(np.isnan(X1)))
        X1 = ma.masked_array(X1, np.isnan(X1))
        X1 = X1[:, ocean]
        X1 = scaler_chl.transform(X1)
        PCs = skpca.transform(X1)
        PCs = PCs[:, :ipc]
    EOFs = skpca.components_
    EOFs = EOFs[:ipc, :]
    EOF_recons = np.ones((ipc, len(lat) * len(lon))) * -999.
    for i in range(ipc): 
        EOF_recons[i,ocean] = EOFs[i, :]
    EOF_recons = ma.masked_values(np.reshape(EOF_recons, (ipc, len(lat), len(lon)), order='F'), -999.)
    varfrac = skpca.explained_variance_ratio_[0:15]*100
    parallels = np.arange(-90,90,30.)
    meridians = np.arange(-180,180,30)
    varfrac = skpca.explained_variance_ratio_[0:15]*100
    for i in range(0, neof):
        fig = plt.figure(figsize=(12, 9))
        plt.subplot(211)
        m = Basemap(projection='cyl', llcrnrlon=min(lon), llcrnrlat=min(lat), urcrnrlon=max(lon), urcrnrlat=max(lat))    
        x, y = m(*np.meshgrid(lon, lat))
        clevs = np.linspace(-2, 2, 40)
        cs = m.contourf(x, y, EOF_recons[i, :, :].squeeze()*100, clevs, cmap=plt.cm.RdBu_r)
        m.drawcoastlines()  
        #m.drawparallels(parallels, labels=[1,0,0,0])
        #m.drawmeridians(meridians, labels=[1,0,0,1])
        cb = m.colorbar(cs, 'right', size='5%', pad='2%')
        cb.set_label('EOF', fontsize=12)
        plt.title('EOF ' + str(i+1) + label+" "+ "{:.2f}%".format(varfrac[i]), fontsize=16)
        plt.subplot(212)
        days = np.linspace(startY, endY, nt)
        daays = np.linspace(startY, endY, ny, dtype=np.int64)
        for item in daays:
            plt.axvline(x=item, color='r', linestyle="-", linewidth=0.7)
        plt.plot(days, PCs[:, i]/100, linewidth=2)
        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.axhline(0, color='k')
        plt.xlabel('Year')
        plt.ylabel('PC Amplitude')   
        plt.ylim(-2, 2)
        if save:
            plt.savefig('./figs/EOF_' + str(i+1) + "_" + label + "_ann")
    plt.figure(figsize=(11, 6))
    eof_num = range(1, 16)
    plt.plot(eof_num, varfrac[0:15], linewidth=2)
    plt.plot(eof_num, varfrac[0:15], linestyle='None', marker="o", color='r', markersize=8)
    plt.axhline(0, color='k')
    plt.xticks(range(1, 16))
    plt.title('Fraction of the total variance represented by each EOF')
    plt.xlabel('EOF #')
    plt.ylabel('Variance Fraction '+label)
    plt.xlim(1, 15)
    plt.ylim(np.min(varfrac), np.max(varfrac) + 0.01)
    if save:
        plt.savefig('./figs/VAR_'+label+"_ann")
    if retrun_eof:
        return PCs, EOF_recons
    else:
        return PCs
