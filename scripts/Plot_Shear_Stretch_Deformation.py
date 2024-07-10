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



datasource = 'imerg'
level = '925hPa'

cluster = ['1','2','3']
#cluster = '2'

dir = 'D:/ERA5/'
outdir = 'D:/00_Final_npj/figures/'

uwnd = xr.open_mfdataset(dir + 'UWND/' + level + '/*.nc')
vwnd = xr.open_mfdataset(dir + 'VWND/' + level + '/*.nc')

# Open clusters data
data = pd.read_csv('D:/00_Final_npj/data/pca_kmeans_optimum_3clusters_' + datasource + '.csv')

for c in cluster:
    cluster_number = c
    num = int(c) - 1

    fname = level + '_STD_Cluster' + cluster_number + '_' + datasource + '.png'

    cluster = data[data['Kmeans_PCA'] == num]
    time_cluster = pd.to_datetime(cluster.Date)

    #filter data according to cluster dates
    cluster_u = uwnd['u'][uwnd.time.isin(time_cluster.values)]
    cluster_v = vwnd['v'][vwnd.time.isin(time_cluster.values)]

    mean_uwnd = np.mean(cluster_u,axis=0)
    mean_vwnd = np.mean(cluster_v,axis=0)

    lat = mean_uwnd['lat'][:]
    lon = mean_uwnd['lon'][:]


    #Smoothing with gaussian-filter
    mean_uwnd = mean_uwnd * units('m/s')
    mean_vwnd = mean_vwnd * units('m/s')

    #total_def = mpcalc.total_deformation(mean_uwnd, mean_vwnd)
    shr_def = mpcalc.shearing_deformation(mean_uwnd, mean_vwnd)
    str_def = mpcalc.stretching_deformation(mean_uwnd, mean_vwnd)

    # Add cyclic point at lon=180
    mean_uwnd, lon1 = add_cyclic_point(mean_uwnd, coord=lon)
    mean_vwnd, lon2 = add_cyclic_point(mean_vwnd, coord=lon)
    shr_def, lon3 = add_cyclic_point(shr_def, coord=lon)
    str_def, lon4 = add_cyclic_point(str_def, coord=lon)

    # Smoothing with gaussian-filter
    uwnd_925 = gaussian_filter(mean_uwnd, sigma=3.0) * units('m/s')
    vwnd_925 = gaussian_filter(mean_vwnd, sigma=3.0) * units('m/s')

    # Plot
    plt.figure(figsize=(10,10))
    projection = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=projection)
    #ax.set_extent([100, -160, 0, 60]) #original
    ax.set_extent([100, 180, 0, 40])
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.yaxis.tick_left()
    ax.set_xticks([100,120,140,160,180], crs=ccrs.PlateCarree())
    ax.set_yticks([10,20,30,40], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_ylim(0,40)

    # streamlines
    ax.streamplot(lon1, lat, uwnd_925, vwnd_925, density=2, linewidth=1, color='black',
                  arrowsize=1, arrowstyle='->', transform=ccrs.PlateCarree())

    norm = mcolors.TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    cf = ax.contourf(lon3, lat, str_def * 1e5, np.arange(-4,4.5,0.5),
                 cmap=plt.cm.PiYG_r,norm=norm,extend='both',transform=ccrs.PlateCarree())

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.06, shrink=0.6, aspect=30)

    cbar.set_ticks(np.arange(-4,5,1))
    cbar.ax.set_ylabel(r'$\mathrm{x10^{-5}\,s^{-1}}$', fontsize=19, weight='bold', rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.2, 1.0)

    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    print('Finished plotting:', c)
    plt.close()