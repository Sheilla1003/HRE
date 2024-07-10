import pandas as pd
import numpy as np
import xarray as xr
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from shapely.geometry import mapping
import geopandas as gpd


# ----------------------------------------------------------------------------------------------------------------------
#DATA USED: IMERG DAILY V07

# This script plots the 24H accumulated precipitation for each clusters
# ----------------------------------------------------------------------------------------------------------------------

plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

dir ='D:/00_Final_npj/'
outdir = dir + 'figures/'

path_shapefile = dir + '/shapefiles/phprov.shp'
shp = gpd.read_file(path_shapefile, crs="epsg:4326")

# Filter only the files at month = 01,02,11,12
months = ['01', '02', '11', '12']
patterns = [f"3B-DAY.MS.MRG.3IMERg.*{year}{month}*.nc4" for year in range(2003, 2023) for month in months]
files = []
for pattern in patterns:
    files.extend(glob.glob(os.path.join(dir + 'data/IMERG/', pattern)))

# Open the dataset
data_imerg = xr.open_mfdataset(files)
precip = data_imerg['precipitation']
precip = precip.sel(time=slice('2003-11-01', '2022-02-28'),lon=slice(116,127),lat=slice(4,21.8))

# Results from the clustering
fname = 'data/pca_kmeans_optimum_3clusters_imerg.csv'

data = pd.read_csv(dir + fname)

cluster_1 = data[data['Kmeans_PCA'] == 0]
cluster_2 = data[data['Kmeans_PCA'] == 1]
cluster_3 = data[data['Kmeans_PCA'] == 2]

clusters = [cluster_1,cluster_2,cluster_3]

i=1
for cluster in clusters:

    date_cluster=pd.to_datetime(cluster.Date)
    print(date_cluster.shape)
    cluster_num = precip.sel(time=date_cluster.values)

    # clipping
    cluster_num.rio.write_crs('epsg:4326', inplace=True)
    cluster_num.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)

    clipped = cluster_num.rio.clip(shp.geometry.apply(mapping), shp.crs, drop=False)

    cluster_mean = np.mean(clipped,axis=0)


    plt.figure(figsize=(10,10))
    projection = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=projection)
    ax.set_xticks([118, 122, 126], crs=ccrs.PlateCarree())
    ax.set_yticks([6, 9, 12, 15, 18, 21], crs=ccrs.PlateCarree())

    shapefile = 'D:/00_Final_npj//shapefiles/phprov.shp'
    shape_feature = ShapelyFeature(Reader(shapefile).geometries(),ccrs.PlateCarree(), facecolor='none')

    # Color scheme of the rainfall data with RGB values as floats
    rgb = [
        [1.0, 1.0, 1.0],  # White
        [0.729, 0.722, 0.722],  # Gray
        [0.0, 0.772, 1.0],  # Light Blue
        [0.420, 0.984, 0.565],  # Light Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.667, 0.0],  # Orange
        [1.0, 0.0, 0.0],  # Red
        [1.0, 0.451, 0.874],  # Pink
        [0.518, 0.0, 0.659]  # Purple
    ]

    cmap = ListedColormap(rgb, "")
    boundaries = [0, 1, 10, 25, 50, 75, 100, 125, 150, 175]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    cf = plt.pcolormesh(clipped.lon, clipped.lat, cluster_mean.transpose(),
                        transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    ax.add_feature(shape_feature,linewidth=0.5)
    #ax.coastlines()
    ax.yaxis.tick_left()
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #plt.tick_params(direction='in')

    cbar = plt.colorbar(cf, ax=ax, pad=0.06, shrink=0.8, aspect=30, extend='max')
    #cbar.ax.tick_params(direction='in')
    cbar.ax.set_title('mm/day',fontsize=20,weight='bold')


    plt.savefig(outdir + 'cluster' + str(i) + '_IMERG.png',bbox_inches='tight', pad_inches=0.1, dpi=600)
    print('Done:','cluster ',i)
    i = i + 1





