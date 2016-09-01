# Copyright (c) 2015,Vienna University of Technology,
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
Module for testing time series to image conversion
'''
import pandas as pd
import numpy as np
import numpy.testing as nptest

from repurpose.ts2img import Ts2Img, agg_tsmonthly
# make a mock read and write class for basic testing of the program logic


class MockGrid(object):

    """
    FakeGrid
    """

    def __init__(self):
        pass

    def grid_points(self):
        """
        return 10 grid points
        """
        return list(range(10)), None, None, None


class MockReader(object):

    """
    Fake Dataset
    """

    def __init__(self, grid):
        self.grid = grid

    def read_ts(self, gpi):
        rng = pd.date_range('1-1-2001', periods=72, freq='D')
        return pd.DataFrame({'data': np.arange(72) + gpi}, index=rng)


class MockWriter(object):

    """
    FakeWriter
    """

    def write_ts(self, gpis, ts):
        assert list(gpis) == list(range(10))
        nptest.assert_almost_equal(ts['data'], np.array([[30, 58], [31, 59],
                                                         [32, 60], [33, 61],
                                                         [34, 62], [35, 63],
                                                         [36, 64], [37, 65],
                                                         [38, 66], [39, 67]]))


def test_ts2img_mock_datasets():
    """
    test the basic programatic logic of the ts2img
    class by using mock datasets that only pass a pandas dataframe
    through
    """

    grid = MockGrid()
    inds = MockReader(grid)
    outds = MockWriter()
    converter = Ts2Img(inds, outds)
    converter.calc()
