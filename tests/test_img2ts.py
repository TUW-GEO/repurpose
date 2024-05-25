# Copyright (c) 2016,Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Vienna University of Technology, Department of
#      Geodesy and Geoinformation nor the names of its contributors may be
#      used to endorse or promote products derived from this software without
#      specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Module for testing image to time series conversion
'''

from datetime import timedelta, datetime
import os
import numpy as np
import pandas as pd
import pytest
from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from pygeobase.object_base import Image
from pygeogrids import BasicGrid, CellGrid
from pygeogrids.netcdf import load_grid
from pynetcf.time_series import OrthoMultiTs, GriddedNcIndexedRaggedTs, GriddedNcOrthoMultiTs
import xarray as xr
from cadati.jd_date import jd2dt
from glob import glob

import tempfile
import numpy.testing as nptest

from repurpose.img2ts import Img2Ts

# make a simple mock Dataset that can be used for testing the conversion


class TestOrthogonalImageDataset(ImageBase):

    def read(self, timestamp=None):

        if timestamp == datetime(2016, 1, 1):
            raise IOError("no data for day")
        # 2x2 pixels around zero lat, lon
        return Image(np.array([0.5, 0.5, -0.5, -0.5]),
                     np.array([1., -1., 1., -1.]),
                     {'var1': np.array([1, 2, 3, 4]) + timestamp.day},
                     {'metavar': 'value'},
                     timestamp=timestamp,
                     timekey=None)

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class TestNonOrthogonalImageDataset(ImageBase):

    def read(self, timestamp=None):

        if timestamp == datetime(2016, 1, 1):
            raise IOError("no data for day")
        jd = pd.to_datetime(timestamp).to_julian_date() - 2400000.5

        # 2x2 pixels around zero lat, lon
        return Image(np.array([0.5, 0.5, -0.5, -0.5]),
                     np.array([1., -1., 1., -1.]),
                     ### GPI 5, 1, 19, 11
                     {'var1': np.array([10, 20, 30, 40]) + timestamp.day,
                      'jd': np.array([jd+0.2, jd+0.1, jd+0.4, jd+0.3])},
                     {'metavar': 'value'},
                     timestamp=timestamp,
                     timekey='jd'
                     )

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class TestMultiTemporalImageDatasetDaily(MultiTemporalImageBase):

    def __init__(self, cls=TestOrthogonalImageDataset):
        super(TestMultiTemporalImageDatasetDaily,
              self).__init__("", cls)

    def tstamps_for_daterange(self, start_date, end_date):
        """
        Return all valid timestamps in a given date range.
        This method must be implemented if iteration over
        images should be possible.

        Parameters
        ----------
        start_date : datetime.date or datetime.datetime
            start date
        end_date : datetime.date or datetime.datetime
            end date

        Returns
        -------
        dates : list
            list of datetimes
        """

        timestamps = []
        diff = end_date - start_date
        for i in range(diff.days + 1):
            daily_date = start_date + timedelta(days=i)
            timestamps.append(daily_date)
        return timestamps


def test_img2ts_nonortho_daily_no_resampling():
    # in this case the output grid is a subset of the input grid
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1., -1., 1., -1.]),
                           gpis=[1, 0, 2, 3])

    target_grid = CellGrid(np.array([0.5, 0.5]),
                           np.array([1., -1.]),
                           gpis=[1, 0],
                           cells=np.array([1, 0]))

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2014, 2, 5)
        end = datetime(2014, 4, 21)

        ds_in = TestMultiTemporalImageDatasetDaily(
            TestNonOrthogonalImageDataset)

        img2ts = Img2Ts(ds_in,
                        outputpath, start, end, imgbuffer=10,
                        input_grid=input_grid,
                        target_grid=target_grid,
                        n_proc=1)
        img2ts.calc()

        assert not img2ts.resample

        ts_should_base = pd.date_range(start, end, freq='D')
        ts_should_jd_gpi0 = ts_should_base + timedelta(days=0.1)
        ts_should_jd_gpi1 = ts_should_base + timedelta(days=0.2)
        ts_should_gpi0 = 20 + ts_should_jd_gpi0.day
        ts_should_gpi1 = 10 + ts_should_jd_gpi1.day
        grid = load_grid(os.path.join(outputpath, 'grid.nc'))

        with GriddedNcIndexedRaggedTs(outputpath, grid=grid) as ds:
            lon, lat = ds.grid.gpi2lonlat(0)
            assert lon == 0.5
            assert lat == -1
            lon, lat = ds.grid.gpi2lonlat(1)
            assert lon == 0.5
            assert lat == 1

            ts = ds.read(0)
            np.testing.assert_array_equal(ts.index.to_julian_date(),
                                          ts_should_jd_gpi0.to_julian_date())
            np.testing.assert_array_equal(ts['var1'].values,
                                          ts_should_gpi0)
            ts = ds.read(1)
            np.testing.assert_array_equal(ts.index.to_julian_date(),
                                          ts_should_jd_gpi1.to_julian_date())
            np.testing.assert_array_equal(ts['var1'].values,
                                          ts_should_gpi1)

            with pytest.raises(IndexError):
                _ = ds.read(2)
            ds.close()

        ds = xr.open_dataset(os.path.join(outputpath, '0000.nc'))

        np.testing.assert_array_equal(ds['location_id'].data,
                                      np.array([0]))
        np.testing.assert_array_equal(ds['lat'].data,
                                      np.array([-1.]))
        np.testing.assert_array_equal(ds['lon'].data,
                                      np.array([0.5]))
        ds.close()
        ds_in.close()


def test_img2ts_nonortho_daily_resampling():
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1., -1., 1., -1.]), )

    target_grid = CellGrid(np.array([0.4, 0.6, -0.4, -0.6]),
                           np.array([0.9, -1.1, 1.1, -0.9]),
                           gpis=[5, 1, 19, 11],
                           cells=[3, 3, 1, 1])

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2014, 2, 5)
        end = datetime(2014, 4, 21)

        ds_in = TestMultiTemporalImageDatasetDaily(
            TestNonOrthogonalImageDataset)

        img2ts = Img2Ts(ds_in, outputpath, start, end, imgbuffer=20,
                        target_grid=target_grid, input_grid=input_grid,
                        r_neigh=4, n_proc=1)
        img2ts.calc()

        assert img2ts.resample is True

        assert len(glob(os.path.join(outputpath, '*.nc'))) == 3

        dates_should = ds_in.tstamps_for_daterange(start, end)
        dates_should = pd.DatetimeIndex(dates_should).to_julian_date() + 0.1
        dates_should = jd2dt(dates_should.values)
        grid = load_grid(os.path.join(outputpath, 'grid.nc'))
        ds = GriddedNcIndexedRaggedTs(outputpath, grid=grid)
        _ = ds.read(5)  # this GPI should have data
        with pytest.raises(OSError):
            ds.read(2)  # this one not, because it wasn't in the target grid
        ts = ds.read(1)  # this one we check later
        ds.close()
        assert np.all(dates_should == ts.index)
        nptest.assert_allclose(ts['var1'], ts.index.day + 20)

        ds_in.close()

        ds = xr.open_dataset(os.path.join(outputpath, '0003.nc'))

        np.testing.assert_array_equal(ds['location_id'].data,
                                      np.array([1, 5]))

        np.testing.assert_array_almost_equal(ds['lat'].data,
                                      np.array([-1.1, 0.9]))

        nptest.assert_allclose(ds['location_id'].data,
                               np.array([1, 5]))
        ds.close()
        ds_in.close()

def test_img2ts_ortho_daily_no_resampling():
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1, -1, 1, -1]), )

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2014, 2, 5)
        end = datetime(2014, 4, 21)

        ds_in = TestMultiTemporalImageDatasetDaily(
            cls=TestOrthogonalImageDataset)
        img2ts = Img2Ts(ds_in, outputpath, start, end, imgbuffer=20,
                        input_grid=input_grid, n_proc=2,
                        cellsize_lat=180, cellsize_lon=360)

        ts_should_gpi0 = pd.date_range(start, end, freq='D').day + 1
        dates_should = pd.DatetimeIndex(ds_in.tstamps_for_daterange(start, end))
        img2ts.calc()

        assert img2ts.resample is False

        ts_file = os.path.join(outputpath, '0000.nc')

        grid = load_grid(os.path.join(outputpath, 'grid.nc'))
        ds = GriddedNcOrthoMultiTs(outputpath, grid)
        ts = ds.read(0)
        assert np.all(dates_should == ts.index)

        with OrthoMultiTs(ts_file) as ds:
            ts = ds.read('var1', 0)
            nptest.assert_allclose(ts['var1'], ts_should_gpi0)
            nptest.assert_allclose(pd.DatetimeIndex(ts['time']).to_julian_date(),
                                   dates_should.to_julian_date())
            nptest.assert_allclose(ds.dataset.variables['location_id'][:],
                                   np.array([0, 1, 2, 3]))
            ds.close()

        ds.close()
        ds_in.close()

def test_img2ts_ortho_daily_resampling():
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1., -1., 1., -1.]),
                           gpis=[0, 1, 2, 3])

    target_grid = BasicGrid(np.array([0.4, 0.6, -0.4, -0.6]),
                            np.array([0.9, -1.1, 1.1, -0.9]))

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2014, 2, 5)
        end = datetime(2014, 4, 21)

        ds_in = TestMultiTemporalImageDatasetDaily(
            cls=TestOrthogonalImageDataset)
        img2ts = Img2Ts(ds_in, outputpath, start, end, imgbuffer=20,
                        target_grid=target_grid, cellsize_lon=360,
                        cellsize_lat=180,
                        input_grid=input_grid, r_neigh=4,
                        n_proc=1)
        img2ts.calc()

        assert img2ts.resample is True

        ts_should_gpi0 = pd.date_range(start, end, freq='D').day + 1
        ts_should_gpi2 = pd.date_range(start, end, freq='D').day + 3
        dates_should = ds_in.tstamps_for_daterange(start, end)

        grid = load_grid(os.path.join(outputpath, 'grid.nc'))
        ds = GriddedNcOrthoMultiTs(outputpath, grid=grid)
        nptest.assert_allclose(ds.read(0)['var1'], ts_should_gpi0)
        nptest.assert_allclose(ds.read(2)['var1'], ts_should_gpi2)

        assert np.all(dates_should == ds.read(0).index)

        ds_in.close()

        ds = xr.open_dataset(os.path.join(outputpath, '0000.nc'))

        np.testing.assert_array_equal(ds['location_id'].data,
                                      np.array([0, 1, 2, 3]))

        np.testing.assert_array_almost_equal(ds['lat'].data,
                                      np.array([0.9, -1.1,  1.1, -0.9]))

        nptest.assert_allclose(ds['location_id'].data,
                               np.array([0, 1, 2, 3]))
        ds.close()
        ds_in.close()

def test_img2ts_ortho_daily_no_resampling_missing_day():
    """
    Test resampling over missing day 2016-01-01 (see reader above)
    """
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1, -1, 1, -1]), )

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2015, 12, 5)
        end = datetime(2016, 1, 10)

        ds_in = TestMultiTemporalImageDatasetDaily(TestOrthogonalImageDataset)
        img2ts = Img2Ts(ds_in,
                        outputpath, start, end, imgbuffer=15,
                        cellsize_lat=180, cellsize_lon=360,
                        input_grid=input_grid, ignore_errors=False)

        img2ts.calc()

        dates = pd.date_range(start, end, freq='D')
        idx_to_drop = np.where(dates == '2016-01-01')[0]
        dates_should = np.delete(dates, idx_to_drop)
        ts_should_gpi0 = dates_should.day + 1

        ts_file = os.path.join(outputpath, '0000.nc')
        with OrthoMultiTs(ts_file) as ds:
            ts = ds.read('var1', 0)
            nptest.assert_allclose(ts['var1'], ts_should_gpi0)
            np.all(dates_should == pd.DatetimeIndex(ts['time']))
            nptest.assert_allclose(ds.dataset.variables['location_id'][:],
                                   np.array([0, 1, 2, 3]))
