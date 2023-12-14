'''
Integration tests for ts2img module
'''
import tempfile
import pandas as pd
import numpy as np
import pytest
import xarray as xr
import os

from repurpose.ts2img import Ts2Img
from datetime import timedelta
from pygeogrids.grids import genreg_grid, CellGrid
from smecv_grid.grid import SMECV_Grid_v052

bbox_eu = (-10, 33, 40, 64)


class DummyReader:
    def __init__(self, timestamps, bbox=bbox_eu, perc_nan=0.4, dropna=False,
                 dates_force_empty=None, drop_time=False):
        self.index = timestamps
        self.grid: CellGrid = SMECV_Grid_v052().subgrid_from_bbox(*bbox)
        self.perc_nan = perc_nan
        self.dropna = dropna
        self.dates_force_empty = dates_force_empty
        self.drop_time = drop_time

    def read(self, lon: float, lat: float):
        """
        - random missing indices
        - random time stamps
        """
        gpi, dist = self.grid.find_nearest_gpi(lon, lat)
        if dist > 25000:
            return None
        rng = np.random.default_rng(seed=int(lon * 100 + lat * 100))

        index = self.index.to_pydatetime()
        i_miss = []
        for d in self.dates_force_empty:
            i_miss += np.where(index == d)[0].tolist()

        timeoffset = [timedelta(seconds=int(s)) for s in
                      rng.choice(np.arange(0, 6 * 60 * 60), len(index))]
        d1 = rng.random(len(index))
        d2 = rng.integers(0, 100, len(index))
        if self.perc_nan:
            idx = rng.choice(np.arange(len(index)),
                             int(len(index) * self.perc_nan),
                             replace=False)
            d1[idx] = np.nan
            idx = rng.choice(np.arange(len(index)),
                             int(len(index) * self.perc_nan),
                             replace=False)
            d2[idx] = -9999
        data = {'var0': d1, 'var2': d2}
        if not self.drop_time:
            index = index + timeoffset

        df = pd.DataFrame(index=index, data=data).sort_index()

        df.iloc[i_miss, :] = np.nan

        if self.dropna:
            df = df.dropna(how='any')
        else:
            df = df.dropna(how='all')

        return df


def test_ts2img_time_collocation_integration():
    def preprocess_func(df, mult=2):
        # This dummy function just adds a new column to the dataframe after
        # reading
        df['var3'] = df['var1'] * mult
        return df

    def postprocess_func(stack, vars, fillvalue=0):
        # This dummy function just fills nans with an actual value before
        # writing the stack
        for var in vars:
            stack[var].values = np.nan_to_num(stack[var].values, nan=fillvalue)
        return stack

    timestamps_image = pd.date_range('2020-07-01', '2020-07-31', freq='6H')
    timestamps_ts = timestamps_image[20:50]

    # 2020070412 and 2020070418 are missing:
    dates_force_empty = [timestamps_ts[14], timestamps_ts[15]]
    reader = DummyReader(timestamps_ts, dropna=False,
                         dates_force_empty=dates_force_empty,
                         perc_nan=0.4)
    # Grid Italy
    img_grid = genreg_grid(0.5, 0.5, 40, 45, 10, 14, origin="bottom")

    _ = reader.read(15, 45)
    # second and last time stamp is missing for testing
    converter = Ts2Img(reader, img_grid, timestamps=timestamps_image,
                       max_dist=25000, time_collocation=True,
                       variables={'var0': 'var1', 'var2': 'var2'})

    with tempfile.TemporaryDirectory() as path_out:
        with pytest.warns(UserWarning):  # expected warning about empty stack
            converter.calc(
                path_out, format_out='slice',
                fn_template="test_{datetime}.nc", drop_empty=True,
                preprocess=preprocess_func, preprocess_kwargs={'mult': 2},
                postprocess=postprocess_func, postprocess_kwargs={'vars': ('var2',)},
                encoding={'var1': {'dtype': 'int64', 'scale_factor': 0.0000001,
                                   '_FillValue': -9999}, },
                var_attrs={'var1': {'long_name': 'test_var1', 'units': 'm'}},
                glob_attrs={'test': 'test'}, var_fillvalues={'var2': -9999},
                var_dtypes={'var2': 'int32'}, n_proc=1)

        assert len(os.listdir(os.path.join(path_out, '2020'))) == 28
        assert os.path.isfile(
            os.path.join(path_out, '2020', 'test_20200708060000.nc'))
        assert os.path.isfile(
            os.path.join(path_out, '2020', 'test_20200712120000.nc'))
        # missing because before the image time stamps
        assert not os.path.isfile(
            os.path.join(path_out, '2020', 'test_202007011200.nc'))
        # missing because forced to be empty
        assert not os.path.isfile(
            os.path.join(path_out, '2020', 'test_202007091200.nc'))
        assert not os.path.isfile(
            os.path.join(path_out, '2020', 'test_202007091800.nc'))

        ds = xr.open_dataset(
            os.path.join(path_out, '2020', 'test_20200708060000.nc'))
        assert list(ds.dims) == ['lon', 'lat', 'time']
        assert ds.data_vars.keys() == {'timedelta_seconds', 'var1', 'var2', 'var3'}

        # var1 was stored as int, but is float64 after decoding
        assert ds['var1'].values.dtype == 'float64'
        assert ds['var1'].values.shape == (1, 10, 8)
        assert ds['var1'].encoding['scale_factor'] == 0.0000001
        # during preprocessing var3 was added as var1 * 2
        np.testing.assert_almost_equal(ds['var3'].values,
                                       ds['var1'].values * 2)
        assert 1 > np.nanmin(ds['var1'].values) > 0
        assert np.isnan(ds['var1'].values[-1, -1, -1])
        np.testing.assert_almost_equal(ds['var1'].values[0, 0, 0], 0.7620138,
                                       5)
        # check if the postprocess function was applied
        assert np.count_nonzero(np.isnan(ds['var2'].values)) == 0

        t = pd.to_datetime(ds.time.values[0]).to_pydatetime()
        t = t + timedelta(seconds=int(
            ds.sel(lon=11.25, lat=44.75)['timedelta_seconds'].values[0]))
        val_ts = reader.read(11.25, 44.75).loc[t]

        val = ds['var1'].sel(lon=11.25, lat=44.75).values[0]
        np.testing.assert_almost_equal(val, val_ts['var0'], decimal=5)

        assert ds['var2'].values.dtype == 'int32'
        assert ds['var2'].values.shape == (1, 10, 8)
        assert 'scale_factor' not in ds['var2'].encoding
        assert np.nanmin(ds['var2'].values) == -9999
        assert np.nanmax(ds['var2'].values) < 100
        val = ds['var2'].sel(lon=11.25, lat=44.75).values[0]
        assert val == int(val_ts['var2'])

        ds.close()   # needed on Windows!



def test_ts2img_no_collocation_integration():
    def preprocess_func(df, **kwargs):
        df.replace(-9999, np.nan, inplace=True)
        df = df.reindex(pd.date_range('2020-07-01', '2020-07-10', freq='1D'))
        df = df.resample('1D').mean()
        df['var3'] = np.nan
        df.loc['2020-07-10', 'var3'] = 1
        df.loc['2020-07-09', 'var3'] = 2
        return df

    def postprocess_func(stack, **kwargs):
        stack = stack.assign(var4=lambda x: x['var3'] ** 2)
        return stack

    timestamps_image = pd.date_range('2020-07-01', '2020-07-10', freq='1D')
    timestamps_ts = timestamps_image[1:]

    # 20200701 and 20200704 are missing:
    dates_force_empty = [timestamps_ts[2]]

    reader = DummyReader(timestamps_ts, dropna=False,
                         dates_force_empty=dates_force_empty,
                         perc_nan=0.4, drop_time=True)
    # Grid Italy
    img_grid = genreg_grid(0.25, 0.25, 40, 45, 10, 14,
                           origin="bottom")

    _ = reader.read(15, 45)
    # second and last time stamp is missing for testing
    converter = Ts2Img(reader, img_grid, timestamps=timestamps_image,
                       max_dist=0, time_collocation=False)

    with tempfile.TemporaryDirectory() as path_out:
        converter.calc(path_out, format_out='slice',
                       preprocess=preprocess_func, postprocess=postprocess_func,
                       fn_template="test_{datetime}.nc", drop_empty=False,
                       encoding={'var2': {'dtype': 'int16'}, },
                       var_attrs={
                           'var2': {'long_name': 'test_var2', 'units': 'm'}},
                       glob_attrs={'test': 'test2'},
                       var_fillvalues={'var2': -9999},
                       var_dtypes={'var2': 'int32'}, n_proc=1)

        # all 10 files must exist, first two emtpy
        assert len(os.listdir(os.path.join(path_out, '2020'))) == 10
        # check empty file
        ds = xr.open_dataset(
            os.path.join(path_out, '2020', 'test_20200701000000.nc'))
        ds2 = xr.open_dataset(
            os.path.join(path_out, '2020', 'test_20200704000000.nc'))
        ds3 = xr.open_dataset(
            os.path.join(path_out, '2020', 'test_20200710000000.nc'))
        ds4 = xr.open_dataset(
            os.path.join(path_out, '2020', 'test_20200709000000.nc'))
        assert np.nanmax(ds3['var3'].values) == 1
        assert np.nanmax(ds4['var3'].values) == 2
        for var in ['var0', 'var2']:
            assert np.all(np.nan_to_num(ds[var].values, nan=-1) ==
                          np.nan_to_num(ds2[var].values, nan=-1))

        # check if the postprocessing function was applied
        assert np.nanmax(ds3['var4'].values) == 1
        assert np.nanmax(ds4['var4'].values) == 4
        assert 4 in np.unique(ds4['var4'].values)
        assert 1 in np.unique(ds3['var4'].values)
        assert len(np.unique(ds3['var4'].values)) == \
               len(np.unique(ds4['var4'].values)) == 2

        ds2.close()  # needed on windows!
        ds3.close()
        ds4.close()

        assert list(ds.dims) == ['lon', 'lat', 'time']
        assert 'timedelta_seconds' not in ds.data_vars.keys()
        assert np.all(np.isnan(ds['var0'].values))
        assert np.all(ds['var2'].values == -9999)
        assert ds.data_vars.keys() == {'var0', 'var2', 'var3', 'var4'}

        ds.close()

        ds = xr.open_dataset(
            os.path.join(path_out, '2020', 'test_20200702000000.nc'))
        assert not np.all(np.isnan(ds['var0'].values))
        assert not np.all(ds['var2'].values == -9999)
        # var1 was stored as int, but is float64 after decoding
        assert ds['var0'].values.dtype == 'float32'
        assert ds['var0'].values.shape == (1, 20, 16)
        assert np.isnan(ds['var0'].encoding['_FillValue'])
        assert 'scale_factor' not in ds['var0'].encoding
        assert 1 > np.nanmin(ds['var0'].values) > 0
        assert np.isnan(ds['var0'].values[-1, -1, -1])
        np.testing.assert_almost_equal(ds['var0'].values[0, 0, 0], 0.28230414,
                                       5)

        t = pd.to_datetime(ds.time.values[0]).to_pydatetime()
        val_ts = reader.read(11.125, 44.875).loc[t]

        val = ds['var0'].sel(lon=11.125, lat=44.875).values[0]
        assert np.isnan(val) == np.isnan(val_ts['var0']) == True

        assert ds['var2'].values.dtype == ds['var2'].encoding[
            'dtype'] == 'int16'
        assert ds['var2'].values.shape == (1, 20, 16)
        assert 'scale_factor' not in ds['var2'].encoding
        assert np.nanmin(ds['var2'].values) == -9999
        assert np.nanmax(ds['var2'].values) < 100
        val = ds['var2'].sel(lon=11.125, lat=44.875).values[0]
        assert val == int(val_ts['var2'])

        ds.close()   # needed on Windows!

