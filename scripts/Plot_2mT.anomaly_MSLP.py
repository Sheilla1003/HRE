import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)

from scipy.ndimage import gaussian_filter
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import scipy.stats as stats

plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42


cluster = ['1','2','3']
#cluster = ['3']

dir = 'D:/ERA5/'
outdir = 'D:/00_Final_npj/figures/'


datasource = 'imerg'

mslp = xr.open_mfdataset(dir + 'MSLP/*.nc')
T2m = xr.open_mfdataset(dir + '2m_Temperature/*.nc')
T2m_anomaly = xr.open_mfdataset(dir + '2m_Temp_1991-2022/*.nc')

climat_T2m = T2m_anomaly['t2m']
climat_T2m = climat_T2m.sel(time=climat_T2m['time'].dt.month.isin([1, 2, 11, 12]))
climat_T2m = climat_T2m.sel(time=slice('1991-11-01', '2020-02-29'))
climat_T2m = np.mean(climat_T2m,axis=0)
# Open clusters data
data = pd.read_csv('D:/00_Final_npj/data/pca_kmeans_optimum_3clusters_' + datasource + '.csv')


for c in cluster:
    cluster_number = c

    num = int(c) - 1

    fname = 'v2_T2m_MSLP_Cluster' + cluster_number + '_' + datasource + '.png'

    cluster = data[data['Kmeans_PCA'] == num]
    time_cluster = pd.to_datetime(cluster.Date)

    cluster_mslp = mslp['msl'][mslp.time.isin(time_cluster.values)]
    cluster_T2m = T2m['t2m'][T2m.time.isin(time_cluster.values)]


    mean_mslp = np.mean(cluster_mslp,axis=0)
    mean_T2m = np.mean(cluster_T2m,axis=0)


    T_anomaly = mean_T2m - climat_T2m

    lat = mean_mslp['lat'][:]
    lon = mean_mslp['lon'][:]

    mean_mslp = mean_mslp/100.

    #smoothing
    mean_mslp = gaussian_filter(mean_mslp, sigma=3.0)
    T_anomaly = gaussian_filter(T_anomaly, sigma=3.0)

    #Add cyclic point
    mean_mslp, lon1 = add_cyclic_point(mean_mslp, coord=lon)
    T_anomaly, lon2 = add_cyclic_point(T_anomaly, coord=lon)


    plt.figure(figsize=(10,10))
    projection = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=projection)
    ax.set_extent([100, -160, 0, 60])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.yaxis.tick_left()
    ax.set_xticks([100,120,140,160,180,-160], crs=ccrs.PlateCarree())
    ax.set_yticks([10,20,30,40,50,60], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_ylim(0,60)

    # MSLP contour
    kw_clabels = {'fontsize': 22, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
                  'rightside_up': True, 'use_clabeltext': True}

    clevmslp = np.arange(970., 1034., 2)

    cs1 = ax.contour(lon1, lat, mean_mslp, clevmslp, colors='k', linewidths=0.8,
                     linestyles='solid', transform=ccrs.PlateCarree())
    cs1.collections[11].set_linewidth(2.0)
    ax.clabel(cs1,clevmslp, **kw_clabels,manual=True)

    #Temp anomaly
    clevtemp = np.arange(-8., 9., 1)
    m = ax.contourf(lon2,lat,T_anomaly,clevtemp,cmap='RdBu_r',extend='both',transform=ccrs.PlateCarree())

    cbar = plt.colorbar(m, ax=ax, orientation='horizontal',pad=0.06, shrink=0.8, aspect=30)
    cbar.ax.set_ylabel('K', fontsize=20, weight='bold',rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.1, 1.0)


    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    print('Done',c)
    plt.close()
#plt.show()
