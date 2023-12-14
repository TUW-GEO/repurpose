import numpy as np
import unittest
import os
import pandas as pd
from repurpose.stack import Regular3dimImageStack
import tempfile
import pytest
import xarray as xr
from datetime import timedelta
import warnings
from pygeogrids.grids import genreg_grid

# [minlat, maxlat, minlon, maxlon]
bbox_spain = (36, 45, -10, 1)

class TestRegularImageStackNoCollocation(unittest.TestCase):
    """
    Time series and image stack match exactly (grid and timestamps)
    """
    def setUp(self) -> None:
        self.timestamps = pd.date_range('2020-01-01', '2020-01-02', freq='6H')
        self.writer = Regular3dimImageStack.from_genreg(
            resolution=0.25, extent=bbox_spain, timestamps=self.timestamps,
            time_collocation=False, reference_time="1900-01-01T00:00:00",
            zlib=True)
        self.shape = (int(self.writer.ds['time'].size),
                      int(self.writer.ds['lat'].size),
                      int(self.writer.ds['lon'].size))

    def test_add_3d_data_init(self):
        """
        Initialise a new 3d variable with passed data
        """
        values = np.full(self.shape, 1)
        self.writer.add_variable('var1', values=values, dtype='int32',
                                 attrs={'name': 'variable1'}, static=False)
        assert list(self.writer.ds.data_vars.keys()) == ['var1']
        assert np.all(self.writer.ds['var1'].values == 1)
        assert self.writer.ds['var1'].values.dtype == 'int32'
        assert self.writer.ds['var1'].values.shape[0] > 1
        assert self.writer.ds['var1'].attrs['name'] == 'variable1'

    def test_add_3d_data_via_ts(self):
        """
        Initialise empty variable then add data via time series
        """
        ts = pd.DataFrame(index=self.timestamps, data={'var1': np.arange(5.0)})
        lon, lat = -4.125, 39.375
        gpi, dist = self.writer.grid.find_nearest_gpi(lon, lat)
        assert dist == 0
        self.writer.write_ts(
            ts, gpi, new_var_kwargs=dict(var1=dict(static=False, attrs={'test': 1},
                                         dtype='float64')))

        val_is = self.writer.ds['var1'].sel(
            time=['2020-01-01T00:00:00'], lon=lon, lat=lat, method='nearest')
        assert val_is.values[0] == 0.0
        assert val_is.dtype == 'float64'

        # make sure to warn when trying to add a static variable
        ts = pd.DataFrame(index=self.timestamps, data={'var2': np.arange(5.0)})
        with pytest.warns(UserWarning):
            self.writer.write_ts(ts, gpi, new_var_kwargs=dict(var2=dict(static=True)))

    def test_add_3d_data_via_loc(self):
        """
        Initialise empty variable then add data for a single timestamp via gpis
        """
        lons, lats = [-4.375, -4.125], [39.125, 39.375]
        gpis = self.writer.grid.find_nearest_gpi(lons, lats)[0]
        data = np.array([1., 2.])
        self.writer.write_loc(
            gpis, data={'var1': data},
            timestamp='2020-01-01T00:00:00')
        assert list(self.writer.ds.data_vars.keys()) == ['var1']

        for i, (lon, lat) in enumerate(zip(lons, lats)):
            val_is = self.writer.ds['var1'].sel(
                time=['2020-01-01T00:00:00'], lon=lon, lat=lat,
                method='nearest').values[0]
            assert val_is == data[i]

    def test_add_2d_data_init(self):
        """
        Initialise a new 2d variable with passed data
        """
        shape = self.writer.grid.shape
        data = np.full(shape, 1.)
        self.writer.add_variable('var1', values=data, dtype='int16',
                                 static=True, attrs={'type': 'static'})
        assert self.writer.ds['var1'].values.shape == shape
        assert self.writer.ds['var1'].values.dtype == 'int16'
        assert self.writer.ds['var1'].attrs['type'] == 'static'
        assert np.all(self.writer.ds['var1'].values == 1)

    def test_add_2d_data_loc(self):
        """
        Initialise empty variable then add data for a single timestamp via
        gpis
        """
        lons, lats = [-4.375, -4.125], [39.125, 39.375]
        gpis = self.writer.grid.find_nearest_gpi(lons, lats)[0]
        self.writer.write_loc(gpis, data={'var1': np.array([1., 2.])})
        is_vals = self.writer.ds['var1'].sel(lon=lons, lat=lats).values
        assert is_vals.dtype == 'float32'
        assert is_vals[0, 0] == 1.
        assert is_vals[1, 1] == 2.

    def test_write_netcdf(self):
        self.writer.add_variable('var1', values=np.full(self.shape, 1))

        encoding = {'var1': {'dtype': 'int32', '_FillValue': -9999}}
        with tempfile.TemporaryDirectory() as out_path:
            fname = os.path.join(out_path, 'test.nc')
            self.writer.to_netcdf(fname, encoding=encoding)
            assert os.path.isfile(fname)
            ds = xr.open_dataset(os.path.join(out_path, 'test.nc'))
            assert list(ds.data_vars.keys()) == ['var1']
            ds.close()   # needed on Windows!


class TestRegularImageStackWithCollocation(unittest.TestCase):
    """
    Time series and image stack match exactly (grid and timestamps)
    """
    def setUp(self) -> None:
        grid_spain = genreg_grid(1, 1, *bbox_spain, origin="bottom")
        self.img_timestamps = pd.date_range('2020-01-01', '2020-01-02', freq='6H')
        # 1h, 3.5h, 0.125h, 5.9999h, 0h
        self.offsets = np.array([3600, 3600*3.5, 3600*0.125, 3600*5.9999, 0.])
        self.timeoffsets = np.array([timedelta(seconds=o) for o in self.offsets])

        grid_spain.shape = (np.unique(grid_spain.activearrlat).size,
                            np.unique(grid_spain.activearrlon).size)

        self.writer = Regular3dimImageStack(
            grid=grid_spain, timestamps=self.img_timestamps, time_collocation=True,
            reference_time="1900-01-01T00:00:00", zlib=True)

    def tearDown(self) -> None:
        self.writer.close()   # needed on Windows!

    def test_add_3d_data_via_ts(self):
        """
        Initialise empty variable then add data via time series
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",
                                  category=pd.errors.PerformanceWarning)
            ts = pd.DataFrame(
                index=pd.DatetimeIndex(self.img_timestamps + self.timeoffsets),
                data={'var1': np.arange(1.1, 6.1).astype('float32'),
                      'var2': np.arange(11, 16).astype('int8')}
            )
        lon, lat = -4.44, 39.61
        gpi, dist = self.writer.grid.find_nearest_gpi(lon, lat)
        assert dist > 0  # grids are not identical
        # last timestamp is also in the image stack directly
        assert len(np.intersect1d(ts.index, self.writer.timestamps)) == 1
        self.writer.write_ts(
            ts, gpi=gpi, new_var_kwargs=dict(var1=dict(static=False),
                var2=dict(static=False, values=-255, dtype='int8')))

        ds = self.writer.ds.sel(lon=lon, lat=lat, method='nearest')
        assert np.all(ds['var1'].values == ts['var1'].values)
        assert np.all(ds['var2'].values == ts['var2'].values)
        assert ds['var1'].values.dtype == 'float32'
        assert ds['var2'].values.dtype == 'int8'

        print(ds['timedelta_seconds'].values)
        print(self.offsets)
        assert np.array_equal(ds['timedelta_seconds'].values,
                              self.offsets.astype(np.float32))


    def test_add_2d_data_loc(self):
        """
        Initialise empty variable then add data for a single timestamp via gpis
        """
        lons, lats = [-4.1, -5.1], [39.9, 40.9]
        gpis, dist = self.writer.grid.find_nearest_gpi(lons, lats)
        assert np.all(dist > 0)

        # the closes time stamp would be 2020-01-01 12:00:00
        self.writer.write_loc(gpis, data={'var1': np.array([1., 2.])},
                              timestamp="2020-01-01T13:01:01.110000000")
        ds = self.writer.ds.sel(time="2020-01-01T12:00:00",
                                lon=lons, lat=lats, method='nearest')

        assert ds['var1'].values[0, 0] == 1.
        assert ds['var1'].values[1, 1] == 2.

        dt = ds[self.writer.timedelta_seconds].values
        # we only reach 2 decimal precision here...
        np.testing.assert_almost_equal(dt[0, 0], 3661.11, 2)
        np.testing.assert_almost_equal(dt[1, 1], 3661.11, 2)
        assert len(np.unique(self.writer.ds[self.writer.timedelta_seconds].values)) \
               == 2

