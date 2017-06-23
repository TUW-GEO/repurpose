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
from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from pygeobase.object_base import Image
from pygeogrids import BasicGrid
from pynetcf.time_series import OrthoMultiTs

import tempfile
import numpy.testing as nptest

from repurpose.img2ts import Img2Ts

# make a simple mock Dataset that can be used for testing the conversion


class TestImageDataset(ImageBase):

    def read(self, timestamp=None, additional_kw=None):

        if timestamp == datetime(2016, 1, 1):
            raise IOError("no data for day")
        # 2x2 pixels around zero lat, lon
        return Image(np.array([0.5, 0.5,
                               -0.5, -0.5]),
                     np.array([1, -1,
                               1, -1]),
                     {'var1': np.ones(4) * timestamp.day}, {'kw': additional_kw}, timestamp)

    def write(self, data):
        raise NotImplementedError()

    def flush(self):
        pass

    def close(self):
        pass


class TestMultiTemporalImageDatasetDaily(MultiTemporalImageBase):

    def __init__(self):
        super(TestMultiTemporalImageDatasetDaily,
              self).__init__("", TestImageDataset)

    # def _build_filename(self, timestamp, custom_templ=None,
    #                     str_param=None):
    #     """
    #     Test implemenation that does not actually look for filenames.
    #     """

    #     return ""

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


def test_img2ts_daily_no_resampling():
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1, -1, 1, -1]), )

    outputpath = tempfile.mkdtemp()
    start = datetime(2014, 2, 5)
    end = datetime(2014, 4, 21)

    ds_in = TestMultiTemporalImageDatasetDaily()
    img2ts = Img2Ts(ds_in,
                    outputpath, start, end, imgbuffer=15,
                    input_grid=input_grid)

    ts_should = np.concatenate([np.arange(5, 29, dtype=np.float),
                                np.arange(1, 32, dtype=np.float),
                                np.arange(1, 22, dtype=np.float)])
    dates_should = ds_in.tstamps_for_daterange(start, end)
    img2ts.calc()
    ts_file = os.path.join(outputpath, '0000.nc')
    with OrthoMultiTs(ts_file) as ds:
        ts = ds.read_ts('var1', 0)
        nptest.assert_allclose(ts['var1'], ts_should)
        assert dates_should == list(ts['time'])
        nptest.assert_allclose(ds.dataset.variables['location_id'][:],
                               np.array([0, 1, 2, 3]))


def test_img2ts_daily_no_resampling_missing_day():
    """
    Test resampling over missing day 2016-01-01 (see reader above)
    """
    input_grid = BasicGrid(np.array([0.5, 0.5, -0.5, -0.5]),
                           np.array([1, -1, 1, -1]), )

    outputpath = tempfile.mkdtemp()
    start = datetime(2015, 12, 5)
    end = datetime(2016, 1, 10)

    ds_in = TestMultiTemporalImageDatasetDaily()
    img2ts = Img2Ts(ds_in,
                    outputpath, start, end, imgbuffer=15,
                    input_grid=input_grid)

    ts_should = np.concatenate([np.arange(5, 32, dtype=np.float),
                                np.arange(2, 11, dtype=np.float)])
    dates_should = ds_in.tstamps_for_daterange(start, end)
    dates_should.remove(datetime(2016, 1, 1))
    img2ts.calc()
    ts_file = os.path.join(outputpath, '0000.nc')
    with OrthoMultiTs(ts_file) as ds:
        ts = ds.read_ts('var1', 0)
        nptest.assert_allclose(ts['var1'], ts_should)
        assert dates_should == list(ts['time'])
        nptest.assert_allclose(ds.dataset.variables['location_id'][:],
                               np.array([0, 1, 2, 3]))
