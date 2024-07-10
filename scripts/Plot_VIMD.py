import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from scipy.ndimage import gaussian_filter
from metpy.units import units
import metpy.calc as mpcalc
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42


datasource = 'obs'
cluster = ['1','2','3']
#cluster = ['1']
dir = 'D:/ERA5/'
outdir = 'D:/00_Final_npj/figures/'

VIMD = xr.open_mfdataset(dir + 'mean_VIFD/*.nc')


# Open clusters data
data = pd.read_csv('D:/00_Final_npj/data/pca_kmeans_optimum_3clusters_' + datasource + '.csv')


for c in cluster:

    cluster_number = c

    fname = 'VIMFD_Cluster' + cluster_number + '_' + datasource + '.png'


    num = int(c) - 1

    cluster = data[data['Kmeans_PCA'] == num]
    time_cluster = pd.to_datetime(cluster.Date)

    #filter data according to cluster dates
    #vimd = VIMD['vimd'][VIMD.time.isin(time_cluster.values)]
    vimd = VIMD['mvimd'][VIMD.time.isin(time_cluster.values)]


    vimd = np.mean(vimd, axis=0)
    print(vimd.values.min())

    lat = vimd['lat'][:]
    lon = vimd['lon'][:]


    vimd, lon1 = add_cyclic_point(vimd, coord=lon)

    # Plot
    plt.figure(figsize=(10, 10))
    projection = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=projection)
    # ax.set_extent([100, -120, -10, 50])
    ax.set_extent([125, 145, 0, 30])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.yaxis.tick_left()
    ax.set_xticks([115,125,135, 145], crs=ccrs.PlateCarree())
    ax.set_yticks([10, 20, 30], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_ylim(0, 30)


    cf= ax.contourf(lon1, lat, vimd*1e4, np.arange(-9,5,1), cmap='Spectral', transform=ccrs.PlateCarree(),extend='min')
    #cf = ax.contourf(lon1, lat, vimd*1e4, cmap='rainbow_r', transform=ccrs.PlateCarree(),
    #                 extend='min')
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.6, aspect=20)
    cbar.ax.set_ylabel(r'$\mathrm{kg\,m^{-2}\,s^{-1}}$', fontsize=18, weight='bold', rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.15, 1.0)

    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    plt.close()
