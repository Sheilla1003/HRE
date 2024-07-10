import xarray as xr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")


from metpy.units import units
from metpy.calc import divergence

plt.rcParams["font.family"] = 'Helvetica'
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['pdf.fonttype'] = 42

datasource = 'obs'

outdir = 'D:/00_Final_npj/figures/'

# Open clusters data
data = pd.read_csv('D:/00_Final_npj/data/pca_kmeans_optimum_3clusters_' + datasource + '.csv')

w_levs = ['100','150','200','250','300','400','500','600','700','850','925','1000']

# Function to open and parse datasets
def open_parse_datasets(path):
    return xr.open_mfdataset(path + '*.nc').metpy.parse_cf()

omega_df = []
for lev in w_levs:
    path = f'D:/ERA5/VertVelocity/{lev}hPa/'
    omega_df.append(open_parse_datasets(path))
omega = xr.concat(omega_df, dim='plev')

v_df = []
for lev in w_levs:
    path = f'D:/ERA5/VWND/{lev}hPa/'
    v_df.append(open_parse_datasets(path))
v = xr.concat(v_df, dim='plev')

u_df = []
for lev in w_levs:
    path = f'D:/ERA5/UWND/{lev}hPa/'
    u_df.append(open_parse_datasets(path))
u = xr.concat(u_df, dim='plev')

cluster = ['1','2','3']
#cluster=['3']
for c in cluster:

    cluster_number = c

    num = int(c) - 1

    cluster = data[data['Kmeans_PCA'] == num]

    time_cluster = pd.to_datetime(cluster.Date)


    cluster_omega = omega['w'].sel(time=omega.time.isin(time_cluster),lat=slice(0, 30), lon=slice(120, 127))
    cluster_v = v['v'].sel(time=v.time.isin(time_cluster),lat=slice(0, 30), lon=slice(120, 127))
    cluster_u = u['u'].sel(time=u.time.isin(time_cluster),lat=slice(0, 30), lon=slice(120, 127))

    plev = cluster_u['plev'].values / 100
    lat = cluster_u['lat'].values

    convergence = divergence(cluster_u, cluster_v)

    df_convergence = np.nanmean(convergence, axis=(1, 3))
    df_omega = np.mean(cluster_omega, axis=[1, 3])
    df_u = np.mean(cluster_u, axis=[1, 3])
    df_v = np.mean(cluster_v, axis=[1, 3])


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
    xlabels = ['0\N{degree sign}', '5\N{degree sign}N', '10\N{degree sign}N', '15\N{degree sign}N',
               '20\N{degree sign}N', '25\N{degree sign}N', '30\N{degree sign}N']
    ax.set_xticklabels(xlabels)

    norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=3.5)

    m = ax.contourf(lat,plev,df_convergence*1e5,np.arange(-2.5, 4.0, 0.5), cmap='seismic_r',norm=norm)

    cbar = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.06, shrink=0.6, aspect=20)
    cbar.set_ticks(np.arange(-2.5, 4.5, 1))
    cbar.ax.set_ylabel(r'$\mathrm{x\,10^{-5}\,s^{-1}}$', fontsize=18, weight='bold', rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.2, 1.0)

    lat_mesh,plev_mesh = np.meshgrid(lat[::6],plev)
    Q = ax.quiver(lat_mesh, plev_mesh, df_u[:, ::6], df_v[:, ::6], color='black',scale_units='xy',scale=8.0)
    ax.quiverkey(Q, 0.19, 0.18, 11, r'$\mathrm{11\,m\,s^{-1}}$',coordinates='figure')

    fname = 'ave_120_127E_Horizontal_Divergence_Winds_Cluster' + cluster_number + '_' + datasource + '.png'
    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    plt.close()
    print('Done',cluster_number)

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
    xlabels = ['0\N{degree sign}', '5\N{degree sign}N', '10\N{degree sign}N', '15\N{degree sign}N',
               '20\N{degree sign}N', '25\N{degree sign}N', '30\N{degree sign}N']
    ax.set_xticklabels(xlabels)

    norm_omega = TwoSlopeNorm(vmin=-5.0, vcenter=0, vmax=2.0)
    m = ax.contourf(lat, plev, df_omega*1e1,np.arange(-5.0, 2.5, 0.5), cmap='seismic_r',norm=norm_omega)
    cbar = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.06, shrink=0.6, aspect=20)
    cbar.set_ticks(np.arange(-4.0, 3.0, 1))
    cbar.ax.set_ylabel(r'$\mathrm{x\,10^{-1}\,Pa\,s^{-1}}$', fontsize=18, weight='bold', rotation='horizontal')
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.set_label_coords(1.2, 1.0)

    fname = 'ave_120_127E_Vertical_Velocity_Cluster' + cluster_number + '_' + datasource + '.png'
    plt.savefig(outdir + fname, bbox_inches='tight', pad_inches=0.2, dpi=600)
    plt.close()
    print('Done', cluster_number)

