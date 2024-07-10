import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('TkAgg')
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from scipy.ndimage import gaussian_filter
from metpy.units import units
import metpy.calc as mpcalc
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature

plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42

datasource = 'obs'
level = '200hPa'

cluster = ['1','2','3']
#cluster = ['3']
dir = 'D:/ERA5/'
outdir = 'D:/00_Final_npj/figures/locs_SL/'

uwnd = xr.open_mfdataset(dir + 'UWND/' + level + '/*.nc')
vwnd = xr.open_mfdataset(dir + 'VWND/' + level + '/*.nc')

# Open clusters data
data = pd.read_csv('D:/00_Final_npj/data/pca_kmeans_optimum_3clusters_' + datasource + '.csv')

for c in cluster:

    cluster_number = c

    fname = level + '_Winds_Cluster' + cluster_number + '_' + datasource + '.png'



    num = int(c) - 1

    cluster = data[data['Kmeans_PCA'] == num]
    time_cluster = pd.to_datetime(cluster.Date)

    #filter data according to cluster dates
    cluster_u = uwnd['u'][uwnd.time.isin(time_cluster.values)]
    cluster_v = vwnd['v'][vwnd.time.isin(time_cluster.values)]

    mean_uwnd = np.mean(cluster_u,axis=0)
    mean_vwnd = np.mean(cluster_v,axis=0)

    lat = mean_uwnd['lat'][:]
    lon = mean_uwnd['lon'][:]

    #Add cyclic point at lon=180
    mean_uwnd, lon1 = add_cyclic_point(mean_uwnd, coord=lon)
    mean_vwnd, lon2 = add_cyclic_point(mean_vwnd, coord=lon)

    #Smoothing with gaussian-filter
    uwnd_925 = gaussian_filter(mean_uwnd, sigma=3.0) * units('m/s')
    vwnd_925 = gaussian_filter(mean_vwnd, sigma=3.0) * units('m/s')

    # Use MetPy to calculate the wind speed for colorfill plot, change units to knots from m/s
    sped_925 = mpcalc.wind_speed(uwnd_925, vwnd_925)


    # Plot
    plt.figure(figsize=(10,10))
    projection = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=projection)
    #ax.set_extent([100, -120, -10, 50])
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
    #plt.tick_params(direction='in')

    # Color scheme of the wind speed with RGB values as floats
    rgb = [
        [0.871, 0.871, 0.871],        # gray B0 <1
        [0.584, 0.804, 0.89],         # Light Blue B1 (1-2)
        [0.757, 0.894, 0.769],        # Pale Green B2 (2-3)
        [0.733, 0.863, 0.443],        # Light Green B3 (4-5)
        [0.549, 0.804, 0.227],        # Green B4 (6-8)
        [0.051, 0.671, 0.255],        # Dark Green B5 (9,11)
        [0.984, 0.933, 0.416],        # Yellow B6 (11,14)
        [0.937, 0.812, 0.318],        # Dark Yellow B7 (14,17)
        [0.933, 0.627, 0.337],        # Orange B8 (17,21)
        [0.839, 0.463, 0.188],         # Dark Orange B9 (21)
        [0.859, 0.259, 0.094]         # Red B10
    ]

    cmap = ListedColormap(rgb, "")
    boundaries = [0, 1, 2, 4, 6, 9, 11, 14, 17, 21,24]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N,extend='max')
    ticks = boundaries

    #streamlines
    ax.streamplot(lon1, lat, uwnd_925, vwnd_925, density=2, linewidth=1, color='black',
                  arrowsize=1, arrowstyle='->', transform=ccrs.PlateCarree())

    #wind speed
    cf = ax.pcolormesh(lon1, lat, sped_925,cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',pad=0.06, shrink=0.8, aspect=30,extend='max')
    #cbar.ax.tick_params(direction='in')
    cbar.set_ticks([0, 1, 2, 4, 6, 9, 11, 14, 17, 21, 24])
    cbar.ax.set_ylabel(r'$\mathrm{m\,s^{-1}}$', fontsize=20, weight='bold',rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.15, 1.0)
    #plt.show()
    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    print('Finished plotting:',c)
    plt.close()