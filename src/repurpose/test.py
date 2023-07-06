import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature


np.random.seed(451)
lons = np.random.choice(size=10, a=np.arange(-180, 180, 1))
lats = np.random.choice(np.arange(-90, 90, 1), 10)
data = np.random.rand(10) * 100

ds = xr.Dataset(
    data_vars=dict(
        mydata=('loc', data),
    ),
    coords=dict(
        longitude=('loc', lons),
        latitude=('loc', lats),
    ),
    attrs=dict(description="My data"),
)

ds['longitude'].attrs = dict(units='degrees_east', standard_name='longitude', valid_range=(-180, 180))
ds['latitude'].attrs = dict(units='degrees_north', standard_name='latitude', valid_range=(-90, 90))

#ds.to_netcdf('/tmp/test.nc')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1, projection=crs.Robinson())

ax.set_global()

ax.set_extent((-10.905332578009272, 32.99948776314773, 35.039062758516316, 66.02065070331525), crs=crs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, zorder=1)
ax.add_feature(cfeature.BORDERS, zorder=1)

for lon, lat in [(0, 47), (20, 53), (2,40), (1.6,53)]:
    p = np.random.choice([1, 2, 3, 4, 5, 6,7])
    lons = [lon] + [lon+i for i in range(p)]
    lats = [lat] * (p + 1)
    plt.scatter(lons[0], lats[0], c='k', marker='X', transform=crs.PlateCarree(), lw=0.0001, s=200)
    plt.scatter(lons[0], lats[0], c='white', transform=crs.PlateCarree(), s=70)

    plt.scatter(x=lons[1:], y=lats[1:], c=np.random.rand(p),
                transform=crs.PlateCarree(), edgecolors='k',
                linewidths=1, zorder=2, s=40)


plt.colorbar()


plt.show()
