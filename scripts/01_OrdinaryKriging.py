import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Perform Kriging for rainfall interpolation
# Reconstruct the missing rainfall from the interpolated values
# -----------------------------------------------------------------

topdir = 'D:/00_Final_npj/data/'

# Rainfall data
#df_rr = pd.read_csv(topdir + 'RR_2003-2022.csv', encoding='latin-1')
df_rr = pd.read_csv(topdir + 'test.csv', encoding='latin-1')


# Lat,lon coordinates of the synoptic stations
stn_locs = pd.read_csv(topdir + 'locs_stn.csv')

# Initialize empty dataframe
reconstructed_rr = pd.DataFrame(columns=df_rr.columns)
reconstructed_rr = reconstructed_rr.dropna(axis=1, how='all')

num_rows = df_rr.shape[0]

for row in range(num_rows):

    rr = df_rr.iloc[row]

    #Check if there is missing values
    has_nan = rr.isna().any().any()

    if has_nan == True:

        date = rr.iloc[0]
        print('Date:', date)
        rr = pd.DataFrame(rr)
        rr = rr.drop(rr.index[0])
        rr = rr.rename_axis('Station').reset_index()
        rr.rename(columns={'Station': 'Station', row: 'RR'}, inplace=True)

        # Rainfall data with the corresponding latlon in each station
        data = pd.merge(rr, stn_locs, on='Station')
        data['Date'] = pd.Timestamp(date)
        data = data[['Date', 'Station', 'Lat', 'Lon', 'RR']]

        # Rows with missing rainfall values
        rows_with_nan = data[data.isnull().any(axis=1)]


        # ORDINARY KRIGING

        # Remove the missing values
        temp = data.copy(deep=True)
        temp.dropna(inplace=True)

        gridx = np.arange(116.5, 127, 0.05, dtype='float64')
        gridy = np.arange(4.5, 21.5, 0.05, dtype='float64')

        OK = OrdinaryKriging(temp.Lon.values, temp.Lat.values, temp.RR.values, variogram_model='linear', verbose=False,
                             enable_plotting=False, coordinates_type='geographic')

        grid_RR, ss = OK.execute('grid', gridx, gridy)

        # grid_RR = grid_RR[::-1,:]

        # Find the interpolated values for stations with missing data
        iter = rows_with_nan.shape[0]

        for i in range(iter):
            nan_lat = rows_with_nan.Lat.iloc[i]
            nan_lon = rows_with_nan.Lon.iloc[i]

            lat_index = int((nan_lat - 4.5) / 0.05)
            lon_index = int((nan_lon - 116.5) / 0.05)

            value_at_point = grid_RR[lat_index, lon_index]
            value_at_point = round(value_at_point, 1)

            if value_at_point < 0:
                value_at_point = 0

            nan_index = rows_with_nan.index[i]
            rows_with_nan.loc[nan_index] = rows_with_nan.loc[nan_index].fillna(value_at_point)

        temp = pd.concat([temp, rows_with_nan]).sort_index()
        temp = temp.pivot(index='Date', columns='Station', values='RR')
        temp.reset_index(inplace=True)

        reconstructed_rr = pd.concat([reconstructed_rr, temp])

    else:
        rr = pd.DataFrame(rr).transpose()
        reconstructed_rr = pd.concat([reconstructed_rr, rr])


reconstructed_rr.to_csv(topdir + 'reconstructed_test1.csv', index=False)








# fig, ax = plt.subplots()
# ax.plot(nan_lon, nan_lat, marker='o', color='red', markersize=5)
#path_shapefile = 'C:/Users/admin/PycharmProjects/Interpolator/Provinces.shp'
#shapefile = gpd.read_file(path_shapefile)
#fig, ax = plt.subplots()
#shapefile.plot(ax=ax, facecolor='none', edgecolor='black')
#plt.imshow(grid_RR, extent=(116.5, 127, 4.5, 21.5))
#plt.show()
#plt.savefig(topdir + 'check.png')
#ax.imshow(grid_RR)
