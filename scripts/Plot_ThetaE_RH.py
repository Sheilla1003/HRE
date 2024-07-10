import xarray as xr
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter


from metpy.units import units
from metpy.calc import dewpoint_from_relative_humidity
from metpy.calc import equivalent_potential_temperature
from metpy.calc import vertical_velocity



plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42

datasource = 'obs'

cluster = ['1','2','3']

outdir = 'D:/00_Final_npj/figures/'


# Open clusters data
data = pd.read_csv('D:/00_Final_npj/data/pca_kmeans_optimum_3clusters_' + datasource + '.csv')


w_levs = ['100','150','200','250','300','400','500','600','700','850','925','1000']

# Function to open and parse datasets
def open_parse_datasets(path):
    return xr.open_mfdataset(path + '*.nc').metpy.parse_cf()

rh_df = []
for lev in w_levs:
    path = f'D:/ERA5/RH/{lev}hPa/'
    rh_df.append(open_parse_datasets(path))
rh = xr.concat(rh_df, dim='plev')

temp_df = []
for lev in w_levs:
    path = f'D:/ERA5/TEMPERATURE/{lev}hPa/'
    temp_df.append(open_parse_datasets(path))
temp = xr.concat(temp_df, dim='plev')


#First:120,130
rh = rh.sel(lat=slice(0, 30),lon=slice(120,127))
temp = temp.sel(lat=slice(0, 30),lon=slice(120,127))


cluster = ['1','2','3']

for c in cluster:

    cluster_number = c

    fname = 'ave_120_127E_RH_ThetaE_Cluster' + cluster_number + '_' + datasource + '.png'

    num = int(c) - 1

    cluster = data[data['Kmeans_PCA'] == num]

    time_cluster = pd.to_datetime(cluster.Date)

    cluster_rh = rh['r'].sel(time=rh.time.isin(time_cluster))
    cluster_temp = temp['t'].sel(time=temp.time.isin(time_cluster))


    td = dewpoint_from_relative_humidity(cluster_temp*units.K , cluster_rh*units.percent)
    theta_e = equivalent_potential_temperature(cluster_rh.plev*units.Pa,cluster_temp, td)

    df_rh = np.mean(cluster_rh, axis=[1, 3])
    df_theta_e = np.mean(theta_e, axis=[1, 3])
    #df_omega = np.mean(cluster_omega, axis=[1, 3])
    #df_v = np.mean(cluster_v, axis=[1, 3])
    plev = cluster_rh['plev'].values / 100
    lat = cluster_rh['lat'].values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.invert_yaxis()
    plt.yscale('log')
    ax.set_ylabel('Pressure (hPa)')
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    subs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    loc = ticker.LogLocator(base=10., subs=subs)
    ax.yaxis.set_major_locator(loc)
    fmt = ticker.FormatStrFormatter("%g")
    ax.yaxis.set_major_formatter(fmt)
    #ax.set_ylim(1000,100)
    #ax.set_xlim(0,30)
    xlabels = ['0\N{degree sign}', '5\N{degree sign}N', '10\N{degree sign}N', '15\N{degree sign}N',
               '20\N{degree sign}N', '25\N{degree sign}N', '30\N{degree sign}N']
    ax.set_xticklabels(xlabels)

    df_rh = gaussian_filter(df_rh, sigma=1.0)

    clevRH = np.arange(0, 110, 10)
    m = ax.contourf(lat, plev, df_rh, clevRH, cmap='jet')
    cbar = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.06, shrink=0.8, aspect=30)
    cbar.ax.set_ylabel('%', fontsize=20, weight='bold', rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.05, 1.0)

    kw_clabels = {'fontsize': 20, 'inline': True, 'inline_spacing': 5, 'fmt': '%i',
                  'rightside_up': True, 'use_clabeltext': True}


    df_theta_e = gaussian_filter(df_theta_e, sigma=2.0)
    clevels = np.arange(280, 390, 2)
    cs2 = ax.contour(lat, plev, df_theta_e, clevels, colors='k', linewidths=0.8, linestyles='solid')
    label_levels = clevels[::2]

    plt.clabel(cs2, levels=label_levels, **kw_clabels)


    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    print('Finish:', cluster_number)
    #plt.show()