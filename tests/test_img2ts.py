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
from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from pygeobase.object_base import Image
from pygeogrids import BasicGrid
from pygeogrids.netcdf import load_grid
from pynetcf.time_series import OrthoMultiTs, GriddedNcIndexedRaggedTs, GriddedNcOrthoMultiTs
from glob import glob
import xarray as xr
import pytest

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
                     {'var1': np.ones(4) * timestamp.day},
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
        jd = pd.to_datetime(timestamp).to_julian_date()
        # 2x2 pixels around zero lat, lon
        return Image(np.array([0.5, 0.5, -0.5, -0.5]),
                     np.array([1., -1., 1., -1.]),
                     {'var1': np.ones(4) * timestamp.day,
                      'jd': np.array([jd-0.1, jd+0, jd+0.1, jd+0.2])},
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
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1., -1., 1., -1.]), )

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2014, 2, 5)
        end = datetime(2014, 4, 21)

        ds_in = TestMultiTemporalImageDatasetDaily(
            TestNonOrthogonalImageDataset)

        img2ts = Img2Ts(ds_in,
                        outputpath, start, end, imgbuffer=10,
                        input_grid=input_grid, n_proc=2)
        img2ts.calc()

        ts_should_base = pd.date_range(start, end, freq='D')
        ts_should_jd_gpi0 = ts_should_base + timedelta(days=-0.1)
        ts_should_jd_gpi3 = ts_should_base + timedelta(days=0.2)
        ts_should_var1 = np.array([ds_in.read(d).data['var1'][0]
                                   for d in pd.date_range(start, end)])
        grid = load_grid(os.path.join(outputpath, 'grid.nc'))

        with GriddedNcIndexedRaggedTs(outputpath, grid=grid) as ds:
            lon, lat = ds.grid.gpi2lonlat(0)
            assert lon == 0.5
            assert lat == 1
            lon, lat = ds.grid.gpi2lonlat(2)
            assert lon == -0.5
            assert lat == 1

            np.testing.assert_array_equal(ds.read(0).index.to_julian_date(),
                                          ts_should_jd_gpi0.to_julian_date())
            np.testing.assert_array_equal(ds.read(0)['var1'].values,
                                          ts_should_var1)
            np.testing.assert_array_equal(ds.read(3).index.to_julian_date(),
                                          ts_should_jd_gpi3.to_julian_date())

            ds.close()

        ds = xr.open_dataset(os.path.join(outputpath, '0000.nc'))

        np.testing.assert_array_equal(ds['location_id'].data,
                                      np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(ds['lat'].data,
                                      np.array([1., -1., 1., -1.]))
        np.testing.assert_array_equal(ds['lon'].data,
                                      np.array([0.5, 0.5, -0.5, -0.5]))
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
                        input_grid=input_grid, n_proc=4)

        ts_should = np.concatenate([np.arange(5, 29, dtype=float),
                                    np.arange(1, 32, dtype=float),
                                    np.arange(1, 22, dtype=float)])
        dates_should = ds_in.tstamps_for_daterange(start, end)
        img2ts.calc()
        ts_file = os.path.join(outputpath, '0000.nc')

        grid = load_grid(os.path.join(outputpath, 'grid.nc'))
        ds = GriddedNcOrthoMultiTs(outputpath, grid)
        ts = ds.read(0)
        assert np.all(dates_should == ts.index)

        with OrthoMultiTs(ts_file) as ds:
            ts = ds.read('var1', 0)
            nptest.assert_allclose(ts['var1'], ts_should)
            for i, t in enumerate(ts['time'].data):
                assert dates_should[i] == t
            nptest.assert_allclose(ds.dataset.variables['location_id'][:],
                                   np.array([0, 1, 2, 3]))
            ds.close()

        ds.close()
        ds_in.close()

def test_img2ts_ortho_daily_resampling():
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1., -1., 1., -1.]), )

    target_grid = BasicGrid(np.array([0.4, 0.6, -0.4, -0.6]),
                            np.array([0.9, -1.1, 1.1, -0.9]))

    with tempfile.TemporaryDirectory() as outputpath:
        start = datetime(2014, 2, 5)
        end = datetime(2014, 4, 21)

        ds_in = TestMultiTemporalImageDatasetDaily(
            cls=TestOrthogonalImageDataset)
        img2ts = Img2Ts(ds_in, outputpath, start, end, imgbuffer=20,
                        target_grid=target_grid,
                        input_grid=input_grid, r_neigh=4,
                        n_proc=2)
        img2ts.calc()

        ts_should = np.concatenate([np.arange(5, 29, dtype=float),
                                    np.arange(1, 32, dtype=float),
                                    np.arange(1, 22, dtype=float)])
        dates_should = ds_in.tstamps_for_daterange(start, end)

        grid = load_grid(os.path.join(outputpath, 'grid.nc'))
        ds = GriddedNcOrthoMultiTs(outputpath, grid=grid)
        ts = ds.read(0)
        ds.close()
        nptest.assert_allclose(ts['var1'], ts_should)
        assert np.all(dates_should == ts.index)

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

        ds_in = TestMultiTemporalImageDatasetDaily()
        img2ts = Img2Ts(ds_in,
                        outputpath, start, end, imgbuffer=15,
                        input_grid=input_grid)

        ts_should = np.concatenate([np.arange(5, 32, dtype=float),
                                    np.arange(2, 11, dtype=float)])
        dates_should = ds_in.tstamps_for_daterange(start, end)
        dates_should.remove(datetime(2016, 1, 1))
        img2ts.calc()
        ts_file = os.path.join(outputpath, '0000.nc')
        with OrthoMultiTs(ts_file) as ds:
            ts = ds.read('var1', 0)
            nptest.assert_allclose(ts['var1'], ts_should)
            assert dates_should == list(ts['time'])
            nptest.assert_allclose(ds.dataset.variables['location_id'][:],
                                   np.array([0, 1, 2, 3]))
