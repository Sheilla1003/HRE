import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
import rasterio
import rasterio.mask
from rasterio.plot import show
from rasterio.transform import Affine
from shapely.geometry import box


plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42

proj = "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +ellps=GRS80 +datum=NAD83 +" \
       "units=m +no_defs "
# ----------------------------------------------------------------------------------------------------------------------
def export_kde_raster(Z, XX, YY, min_x, max_x, min_y, max_y, proj, filename):
    '''Export and save a kernel density raster.'''

    # Get resolution
    xres = (max_x - min_x) / len(XX)
    yres = (max_y - min_y) / len(YY)

    # Set transform
    transform = Affine.translation(min_x - xres / 2, min_y - yres / 2) * Affine.scale(xres, yres)

    # Export array as raster
    with rasterio.open(
            filename,
            mode = "w",
            driver = "GTiff",
            height = Z.shape[0],
            width = Z.shape[1],
            count = 1,
            dtype = Z.dtype,
            crs = proj,
            transform = transform,
    ) as new_dataset:
            new_dataset.write(Z, 1)
# ----------------------------------------------------------------------------------------------------------------------

path = 'D:/00_Final_npj/'

outpath_figs = path + 'figures/'


rr = pd.read_csv(path + 'data/pca_kmeans_optimum_3clusters_obs.csv')
stn_locs = pd.read_csv(path + 'data/locs_stn.csv')

cluster_1 = rr[rr['Kmeans_PCA'] == 0]
cluster_2 = rr[rr['Kmeans_PCA'] == 1]
cluster_3 = rr[rr['Kmeans_PCA'] == 2]


#fname = 'cluster3_obs.png'
fname='RR_2003_2022_obs.png'
#data = cluster_3
data=rr
print(data.shape)
columns_to_drop = data.iloc[:, 56:97].columns
data = data.drop(columns=columns_to_drop, axis=1)
data = data.drop('Date', axis=1)
data = data.mean()
data = data.rename_axis('Station').reset_index()
data.rename(columns={'Station': 'Station', 0: 'RR'}, inplace=True)
data= pd.merge(data, stn_locs, on='Station')

#-------------------------------------------------------------------------------------------------------------------
lat = data.Lat
lon = data.Lon

path_shapefile = path + '/shapefiles/phprov.shp'
PH = gpd.read_file(path_shapefile)
data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(lon,lat))

gridx = np.arange(116, 127, 0.05, dtype='float64')
gridy = np.arange(4, 21.8, 0.05, dtype='float64')

# Kriging'
OK = OrdinaryKriging(lon, lat, data.RR, variogram_model='spherical', verbose=False,
                     enable_plotting=False, coordinates_type='geographic')

grid_RR, ss = OK.execute('grid', gridx, gridy)
plt.imshow(grid_RR,extent=(116,127,4,21.8))
plt.gca().invert_yaxis()
kt.write_asc_grid(gridx, gridy, grid_RR, filename= path + fname+'.asc')

# Export the grid_RR to raster
export_kde_raster(Z=grid_RR, XX=gridx, YY=gridy, min_x=116, max_x=127,
                  min_y=4, max_y=21.8, proj=proj,filename=path+'cluster.tif')

# Open the raster
raster_pk = rasterio.open(path+'cluster.tif')

# Create polygon with extent of raster
poly_shapely = box(*raster_pk.bounds)

# Create a dictionary with needed attributes and required geometry column
attributes_df = {'Attribute': ['name1'], 'geometry': poly_shapely}

# Convert shapely object to a GeoDataFrame
raster_pk_extent = gpd.GeoDataFrame(attributes_df, geometry = 'geometry', crs = proj)

# Create copy of test dataset
rain_test_gdf_pk_krig = data

# Subset the GeoDataFrame by checking which test points are within the raster extent polygon
# If a test point is beyond the extent of training points dataset, the kriging output may not cover that test point
rain_test_gdf_pk_krig = rain_test_gdf_pk_krig[rain_test_gdf_pk_krig.within(raster_pk_extent.geometry.values[0])]

# Create list of XY coordinate pairs for the test points that fall within raster extent polygon
coords_rain_test_pk_krig = [list(xy) for xy in zip(rain_test_gdf_pk_krig["geometry"].x, rain_test_gdf_pk_krig["geometry"].y)]

# Extract raster value at each test point and add the values to the GeoDataFrame
rain_test_gdf_pk_krig["VALUE_Predict"] = [x[0] for x in raster_pk.sample(coords_rain_test_pk_krig)]

# Mask raster to counties shape
out_image_pk, out_transform_pk = rasterio.mask.mask(raster_pk, PH.geometry.values, crop = False)

# ----------------------------------------------------------------------------------------------------------------------
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

fig, ax = plt.subplots(1, figsize = (10, 10))
sc = show(out_image_pk, ax = ax, transform = out_transform_pk, cmap=cmap, norm=norm)
PH.plot(ax = ax, color = 'none', edgecolor='black',linewidth=0.4)
plt.gca().invert_yaxis()
#plt.tick_params(direction='in')
im = sc.get_images()[0]
cbar = fig.colorbar(im, ax=ax,pad=0.06,shrink=0.8,aspect=30,extend='max')
#cbar.ax.tick_params(direction='in')
cbar.ax.set_title('mm/day',fontsize=20,weight='bold')





ax.set_xticks([118, 122, 126])
ax.set_yticks([6,9,12,15,18,21])

ax.set_xticklabels(['118°E', '122°E', '126°E'])
ax.set_yticklabels(['6°N','9°N','12°N','15°N','18°N','21°N'])

raster_pk.close()

plt.savefig(outpath_figs + fname, bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.close()

