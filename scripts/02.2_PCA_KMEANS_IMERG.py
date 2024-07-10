import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import glob
import os
from shapely.geometry import mapping
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------------------------------------------------------------------
#DATA USED: IMERG DAILY V07

# This script does the following:
# 1. Uses IMERG data and determine HRE days when the 50 mm/day precipitation is over 10% of land grids
# 2. Determine the number of PCs
# 3. PCA
# 4. KMEANS
# ----------------------------------------------------------------------------------------------------------------------


dir ='D:/00_Final_npj/'
outdir = dir + 'data/'

path_shapefile = dir + '/shapefiles/phprov.shp'

shape_feature = ShapelyFeature(Reader(path_shapefile).geometries(), ccrs.PlateCarree(), facecolor='none')
shp = gpd.read_file(path_shapefile, crs="epsg:4326")

# Filter only the files at month = 01,02,11,12
months = ['01', '02', '11', '12']
patterns = [f"3B-DAY.MS.MRG.3IMERg.{year}{month}*.nc4" for year in range(2003, 2023) for month in months]
files = []
for pattern in patterns:
    files.extend(glob.glob(os.path.join(dir + 'data/IMERG/', pattern)))

# Open the dataset
data = xr.open_mfdataset(files)
precip = data['precipitation']
#precip = precip.sel(time=precip['time'].dt.month.isin([1, 2, 11, 12]),lon=slice(116,127),lat=slice(5,21))
precip = precip.sel(time=slice('2003-11-01', '2022-02-28'),lon=slice(116,127),lat=slice(5,21))


precip.rio.write_crs('epsg:4326', inplace=True)
precip.rio.set_spatial_dims(x_dim='lon',y_dim='lat',inplace=True)

# Clipped only the land data using PH shapefile
clipped = precip.rio.clip(shp.geometry.apply(mapping),shp.crs,drop=False)

#plt.figure(figsize=(10,10))
#projection = ccrs.PlateCarree(central_longitude=180)
#ax = plt.axes(projection=projection)
#ax.pcolormesh(clipped.lon,clipped.lat,clipped[1,:,:].transpose(),transform=ccrs.PlateCarree())
#ax.add_feature(shape_feature,linewidth=0.5)
#ax.coastlines()
#plt.savefig(dir + 'third.png')
#plt.close()


threshold = 50
mask = clipped > 50

num_grids_land = clipped[0,:,:].notnull().sum().values

num_above_50mm = np.zeros(mask.shape[0])
for i in range(mask.shape[0]):
    num_above_50mm[i] = mask[i,:,:].sum()

percent = (num_above_50mm / num_grids_land) * 100
HRE_index = np.where(percent > 10)[0]
HRE = clipped[HRE_index, :, :]


rr_data = HRE.stack(latlon=('lat', 'lon'))
latlon = rr_data.latlon.values
rr = pd.DataFrame(rr_data)

date = pd.DataFrame(HRE.time.values)
rr.insert(0, 'Date', date)
rr.columns = rr.columns.map(str)
rr.set_index('Date',inplace=True)
rr = rr.dropna(axis=1)

# 1. Standardized the data
scaler = StandardScaler()
scaler.fit(rr)
scaled_data = scaler.transform(rr)


# ---------------------------------------------------------------------------------------------------------------------
# Determine the number of pc components using scree plot
pca= PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
t = pca.explained_variance_ratio_

num_components = np.argmax(pca.explained_variance_ratio_.cumsum() >= 0.8) + 1

pca = PCA(n_components=num_components)
pca.fit(scaled_data)
scores_pca = pca.transform(scaled_data)

#plt.figure(figsize = (10,8))
#plt.plot(range(1,189), pca.explained_variance_ratio_.cumsum(),marker='o',
#        linestyle='--')
#plt.title('Explained Variance by Components')
#plt.xlabel('Number of Components')
#plt.ylabel('Cumulative Explained Variance')
#plt.xticks(np.arange(1, 188, 10))
#plt.margins(x=0.01)
#plt.savefig(dir + 'scree_plot.png')
# ---------------------------------------------------------------------------------------------------------------------

# By elbow method the clustering should be around 2 or 3
kmeans_pca = KMeans(n_clusters=3,init = 'k-means++', random_state = 42)
kmeans_pca.fit(scores_pca)


# Saving the results as csv file
result = rr.rename_axis('Date').reset_index()

data_pca_kmeans = pd.concat([result.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)

data_pca_kmeans['Kmeans_PCA'] = kmeans_pca.labels_
data_pca_kmeans.to_csv(outdir + 'pca_kmeans_optimum_3clusters_imerg.csv', index=False)
