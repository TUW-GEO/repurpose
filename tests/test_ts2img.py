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
from repurpose.ts2img import Ts2Img
from datetime import timedelta
from pygeogrids.grids import CellGrid

np.random.seed(123)

class DummyReader:
    def __init__(self):
        self.index = pd.date_range('2020-02-01', '2020-07-31', freq='6H')

    def read(self, lon: float, lat: float):
        """
        - random missing indices
        - random time stamps
        """
        ind = np.random.choice(np.arange(len(self.index)), int(len(self.index)*0.8))
        index = self.index[ind].to_pydatetime()
        timeoffset = [timedelta(seconds=int(s)) for s in np.random.choice(np.arange(0, 12*60*60), len(ind))]
        data = {'var1': np.random.rand(len(ind)),
                'var2': np.random.random_integers(0, 100, len(ind))}
        df = pd.DataFrame(index=index+timeoffset, data=data).sort_index()

        return df

def test_ts2img():
    reader = DummyReader()

    conv = Ts2Img(reader, ['var1', 'var2'], '2020-07-20', '2020-08-01', resolution=1,
           fill_values={'var2': -1})
    conv.
if __name__ == '__main__':
    test_ts2img()
