# Copyright (c) 2020, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import pynetcf.time_series as nc
import pygeogrids.grids as grids
import repurpose.resample as resamp
import numpy as np
import os
from datetime import datetime
import logging
import pygeogrids
import pygeogrids.netcdf as grid2nc


class Img2TsError(Exception):
    pass


class Img2Ts(object):

    """
    class that uses the read_img iterator of the input_data dataset
    to read all images between startdate and enddate and saves them
    in netCDF time series files according to the given netCDF class
    and the cell structure of the outputgrid

    Parameters
    ----------
    input_dataset : DatasetImgBase like class instance
        must implement a daily_images iterator that yields
        data : dict
            dictionary of numpy arrays that hold the image data for each variable
            of the dataset
        timestamp : exact timestamp of the image
        lon : numpy.array or None
            array of longitudes, if None self.grid will be assumed
        lat : numpy.array or None
            array of latitudes, if None self.grid will be assumed
        jd : numpy.array or None
            array of observation times in julian days, if None all
            observations have the same timestamp
    outputpath : string
        path where to save the time series to
    startdate : date
        date from which the time series should start. Of course images
        have to be available from this date onwards.
    enddate : date
        date when the time series should end. Images should be availabe
        up until this date
    input_kwargs : dict, optional
        keyword arguments which should be used in the read_img method of the
        input_dataset
    input_grid : grid instance as defined in :module:`pytesmo.grids.grid`, optional
        the grid on which input data is stored.
        If not given then the grid of the input dataset will be used.
        If the input dataset has no grid object then resampling to the
        target_grid is performed.
    target_grid : grid instance as defined in :module:`pytesmo.grids.grid`, optional
        the grid on which the time series will be stored.
        If not given then the grid of the input dataset will be used
    imgbuffer : int, optional
        number of days worth of images that should be read into memory before
        a time series is written. This parameter should be chosen so that
        the memory of your machine is utilized. It depends on the daily data
        volume of the input dataset
    variable_rename : dict, optional
        if the variables should have other names than the names that are
        returned as keys in the dict by the daily_images iterator. A dictionary
        can be provided that changes these names for the time series.
    unlim_chunksize : int, optional
        netCDF chunksize for unlimited variables.
    cellsize_lat : float, optional
        if outgrid or input_data.grid are not cell grids then the cellsize
        in latitude direction can be specified here. Default is 1 global cell.
    cellsize_lon : float, optional
        if outgrid or input_data.grid are not cell grids then the cellsize
        in longitude direction can be specified here. Default is 1 global cell.
    r_methods : string or dict, optional
        resample methods to use if resampling is necessary, either 'nn' for nearest
        neighbour or 'custom' for custom weight function. Can also be a dictionary
        in which the method is specified for each variable
    r_weightf : function or dict, optional
        if r_methods is custom this function will be used to calculate the weights
        depending on distance. This can also be a dict with a separate weight function
        for each variable.
    r_min_n : int, optional
        Minimum number of neighbours on the target_grid that are required for a point to be resampled.
    r_radius : float, optional
        resample radius in which neighbours should be searched given in meters
    r_neigh : int, optional
        maximum number of neighbours found inside r_radius to use during resampling. If more are found
        the r_neigh closest neighbours will be used.
    r_fill_values : number or dict, optional
        if given the resampled output array will be filled with this value if no valid
        resampled value could be computed, if not a masked array will be returned
        can also be a dict with a fill value for each variable
    filename_templ : string, optional
        filename template must be a string with a string formatter for the cell number.
        e.g. '%04d.nc' will translate to the filename '0001.nc' for cell number 1.
    gridname : string, optional
        filename of the grid which will be saved as netCDF
    global_attr : dict, optional
        global attributes for each file
    ts_attributes : dict, optional
        dictionary of attributes that should be set for the netCDF time series.
        Can be either a dictionary of attributes that will be set for all variables in input_data
        or a dictionary of dictionaries. In the second case the first dictionary has to have a key
        for each variable returned by input_data and the second level dictionary will be the dictionary of
        attributes for this time series.
    ts_dtype : numpy.dtype or dict of numpy.dtypes
        data type to use for the time series, if it is a dict then a key must exist for each
        variable returned by input_data.
        Default : None, no change from input data
    time_units : string, optional
        units the time axis is given in.
        Default: "days since  1858-11-17 00:00:00" which is modified julian date
        for regular images this can be set freely since the conversion is done
        automatically, for images with irregular timestamp this will be ignored for now
    zlib: boolean, optional
        if True the saved netCDF files will be compressed
        Default: True
    """

    def __init__(self, input_dataset, outputpath, startdate, enddate,
                 input_kwargs={}, input_grid=None, target_grid=None, imgbuffer=100, variable_rename=None,
                 unlim_chunksize=100, cellsize_lat=180.0, cellsize_lon=360.0,
                 r_methods='nn', r_weightf=None, r_min_n=1, r_radius=18000,
                 r_neigh=8, r_fill_values=None, filename_templ='%04d.nc',
                 gridname='grid.nc', global_attr=None, ts_attributes=None,
                 ts_dtypes=None, time_units="days since 1858-11-17 00:00:00", zlib=True):

        self.imgin = input_dataset
        self.zlib = zlib
        if not hasattr(self.imgin, 'grid'):
            self.input_grid = input_grid
        else:
            self.input_grid = self.imgin.grid

        if self.input_grid is None and target_grid is None:
            raise ValueError("Either the input dataset has to have a grid, "
                             "input_grid has to be specified or "
                             "target_grid has to be set")

        self.input_kwargs = input_kwargs

        self.target_grid = target_grid
        if self.target_grid is None:
            self.target_grid = self.input_grid
            self.resample = False
        else:
            # if input and target grid are not equal resampling is required
            if self.input_grid != self.target_grid:
                self.resample = True

        # if the target grid is not a cell grid make it one
        # default is just one cell for the
        # this is just a dirty hack until grids have a method
        # to check their grid type
        if (type(self.target_grid) == grids.BasicGrid or
            (pygeogrids.grids.BasicGrid in type(self.target_grid).__bases__ and
             len(type(self.target_grid).__bases__))) == 1:

            self.target_grid = self.target_grid.to_cell_grid(cellsize_lat=cellsize_lat,
                                                             cellsize_lon=cellsize_lon)
        self.currentdate = startdate
        self.startdate = startdate
        self.enddate = enddate
        self.imgbuffer = imgbuffer
        self.outputpath = outputpath
        self.variable_rename = variable_rename
        self.unlim_chunksize = unlim_chunksize
        self.gridname = gridname

        self.r_methods = r_methods
        self.r_weightf = r_weightf
        self.r_min_n = r_min_n
        self.r_radius = r_radius
        self.r_neigh = r_neigh
        self.r_fill_values = r_fill_values

        # if each image has only one timestamp then we are handling
        # time series of type Orthogonal multidimensional array representation
        # according to the CF conventions
        self.orthogonal = None

        self.filename_templ = filename_templ
        self.global_attr = global_attr
        self.ts_attributes = ts_attributes
        self.ts_dtypes = ts_dtypes
        self.time_units = time_units
        self.non_ortho_time_units = "days since  1858-11-17 00:00:00"

    def calc(self):
        """
        go through all images and retrieve a stack of them
        then go through all grid points in cell order and write to netCDF file
        """
        # save grid information in file
        grid2nc.save_grid(
            os.path.join(self.outputpath, self.gridname), self.target_grid)

        for img_stack_dict, start, end, dates, jd_stack in self.img_bulk():
            #==================================================================
            start_time = datetime.now()

            for cell in self.target_grid.get_cells():

                cell_gpis, cell_lons, cell_lats = self.target_grid.grid_points_for_cell(
                    cell)

                # look where in the subset the data is
                cell_index = np.where(
                    cell == self.target_grid.activearrcell)[0]

                if cell_index.size == 0:
                    raise Img2TsError('cell not found in grid subset')

                data = {}

                for key in img_stack_dict:
                    # rename variable in output dataset
                    if self.variable_rename is None:
                        var_new_name = str(key)
                    else:
                        var_new_name = self.variable_rename[key]

                    output_array = np.swapaxes(
                        img_stack_dict[key][:, cell_index], 0, 1)

                    # change dtypes of output time series
                    if self.ts_dtypes is not None:
                        if type(self.ts_dtypes) == dict:
                            output_dtype = self.ts_dtypes[key]
                        else:
                            output_dtype = self.ts_dtypes
                        output_array = output_array.astype(output_dtype)

                    data[var_new_name] = output_array

                if self.orthogonal:

                    with nc.OrthoMultiTs(
                        os.path.join(self.outputpath,
                                     self.filename_templ % cell),
                            n_loc=cell_gpis.size, mode='a',
                            zlib=self.zlib,
                            unlim_chunksize=self.unlim_chunksize,
                            time_units=self.time_units) as dataout:

                        # add global attributes to file
                        if self.global_attr is not None:
                            for attr in self.global_attr:
                                dataout.add_global_attr(
                                    attr, self.global_attr[attr])

                        dataout.add_global_attr(
                            'geospatial_lat_min', np.min(cell_lats))
                        dataout.add_global_attr(
                            'geospatial_lat_max', np.max(cell_lats))
                        dataout.add_global_attr(
                            'geospatial_lon_min', np.min(cell_lons))
                        dataout.add_global_attr(
                            'geospatial_lon_max', np.max(cell_lons))

                        dataout.write_ts_all_loc(cell_gpis, data, dates,
                                                 lons=cell_lons,
                                                 lats=cell_lats,
                                                 attributes=self.ts_attributes)
                elif not self.orthogonal:

                    with nc.IndexedRaggedTs(os.path.join(self.outputpath,
                                                         self.filename_templ %
                                                         cell),
                                            n_loc=cell_gpis.size, mode='a',
                                            zlib=self.zlib,
                                            unlim_chunksize=self.unlim_chunksize,
                                            time_units=self.non_ortho_time_units) as dataout:

                        # add global attributes to file
                        if self.global_attr is not None:
                            for attr in self.global_attr:
                                dataout.add_global_attr(
                                    attr, self.global_attr[attr])

                        dataout.add_global_attr(
                            'geospatial_lat_min', np.min(cell_lats))
                        dataout.add_global_attr(
                            'geospatial_lat_max', np.max(cell_lats))
                        dataout.add_global_attr(
                            'geospatial_lon_min', np.min(cell_lons))
                        dataout.add_global_attr(
                            'geospatial_lon_max', np.max(cell_lons))

                        # for this dataset we have to loop through the gpis since each time series
                        # can be different in length
                        for i, (gpi, gpi_lon, gpi_lat) in enumerate(zip(cell_gpis, cell_lons, cell_lats)):
                            gpi_data = {}
                            # convert to modified julian date
                            gpi_jd = jd_stack[:, cell_index[i]] - 2400000.5
                            # remove measurements that were filled with the fill value
                            # during resampling
                            # doing this on the basis of the time variable should
                            # be enought since without time -> no valid
                            # observations
                            if self.resample:
                                if self.r_fill_values is not None:
                                    if type(self.r_fill_values) == dict:
                                        time_fill_value = self.r_fill_values[
                                            self.time_var]
                                    else:
                                        time_fill_value = self.r_fill_values

                                    valid_mask = gpi_jd != time_fill_value
                                else:
                                    valid_mask = np.invert(gpi_jd.mask)
                                gpi_jd = gpi_jd[valid_mask]
                            else:
                                # all are valid if no resampling took place
                                valid_mask = slice(None, None, None)
                            for key in data:
                                gpi_data[key] = data[key][i, valid_mask]

                            if gpi_jd.data.size > 0:
                                dataout.write_ts(gpi, gpi_data, gpi_jd,
                                                 lon=gpi_lon, lat=gpi_lat,
                                                 attributes=self.ts_attributes,
                                                 dates_direct=True)

            data = {}
            output_array = None
            logging.log(logging.INFO, datetime.now() - start_time)

    def img_bulk(self):
        """
        Yields numpy array of self.const.imgbuffer images,
        start and enddate until all dates have been read

        Returns
        -------
        img_stack_dict : dict of numpy.array
            stack of daily images for each variable
        startdate : date
            date of first image in stack
        enddate : date
            date of last image in stack
        datetimestack : np.array
            array of the timestamps of each image
        jd_stack : np.array or None
            if None all observations in an image have the same
            observation timestamp. Otherwise it gives the julian date
            of each observation in img_stack_dict
        """

        img_dict = {}
        datetimes = []
        jd_list = []
        # set start of current imgbulk to startdate
        bulkstart = self.startdate
        # image counter
        read_images = 0

        dates = self.imgin.tstamps_for_daterange(self.startdate,
                                                 self.enddate)
        for date in dates:
            try:
                (input_img, metadata,
                 image_datetime, lon,
                 lat, time_arr) = self.imgin.read(date, **self.input_kwargs)
            except IOError as e:
                msg = "I/O error({0}): {1}".format(e.errno,
                                                   e.strerror)
                logging.log(logging.INFO, msg)
                continue
            read_images += 1
            logging.log(logging.INFO, "read" + image_datetime.isoformat())
            if self.resample:

                if time_arr is not None:
                    input_img['jd'] = time_arr
                input_img = resamp.resample_to_grid(input_img, lon, lat,
                                                    self.target_grid.activearrlon,
                                                    self.target_grid.activearrlat,
                                                    methods=self.r_methods,
                                                    weight_funcs=self.r_weightf,
                                                    min_neighbours=self.r_min_n,
                                                    search_rad=self.r_radius,
                                                    neighbours=self.r_neigh,
                                                    fill_values=self.r_fill_values)
                time_arr = input_img.pop('jd')
            if time_arr is None:
                self.time_var = None

            else:
                self.time_var = 'jd'
            if time_arr is not None:
                # if time_var is not None means that each observation of the
                # image has its own observation time
                # this means that the resulting time series is not
                # regularly spaced in time
                if self.orthogonal is None:
                    self.orthogonal = False
                if self.orthogonal:
                    raise Img2TsError("Images can not switch between a fixed image "
                                      "timestamp and individual timestamps for each observation")
                jd_list.append(time_arr)
            if time_arr is None:
                if self.orthogonal is None:
                    self.orthogonal = True
                if not self.orthogonal:
                    raise Img2TsError(
                        "Images can not switch between a fixed image "
                        "timestamp and individual timestamps for each observation")

            for key in input_img:
                if key not in img_dict.keys():
                    img_dict[key] = []
                img_dict[key].append(input_img[key])

            datetimes.append(image_datetime)

            if read_images >= self.imgbuffer - 1:
                img_stack_dict = {}
                if len(jd_list) != 0:
                    jd_stack = np.ma.vstack(jd_list)
                    jd_list = None
                else:
                    jd_stack = None
                for key in img_dict:
                    img_stack_dict[key] = np.vstack(img_dict[key])
                    img_dict[key] = None
                datetimestack = np.array(datetimes)
                img_dict = {}
                datetimes = []
                jd_list = []
                yield (img_stack_dict, bulkstart, self.currentdate,
                       datetimestack, jd_stack)
                # reset image counter
                read_images = 0

        if len(datetimes) > 0:
            img_stack_dict = {}
            if len(jd_list) != 0:
                jd_stack = np.ma.vstack(jd_list)
            else:
                jd_stack = None
            for key in img_dict:
                img_stack_dict[key] = np.vstack(img_dict[key])
                img_dict[key] = None
            datetimestack = np.array(datetimes)
            yield (img_stack_dict, bulkstart, self.currentdate, datetimestack,
                   jd_stack)


if __name__ == '__main__':
    import rsdata.GLDAS_NOAH.interface as GLDAS
    import rsdata.root_path as root

    # outputpath = os.path.join(root.d, 'GIO GL', 'Evolution', 'SWI_NRT_validation', 'GLDAS',
    #                           'raw')
    outputpath = os.path.join(root.d, 'img2ts_test')
    start = datetime(2014, 2, 5)
    end = datetime(2014, 4, 21)

    img2ts = Img2Ts(GLDAS.GLDAS025Img(parameter=['086_L1', '086_L2',
                                                 '086_L3', '086_L4', '131',
                                                 '132', '138', '085_L1',
                                                 '065', '057']),
                    outputpath, start, end, imgbuffer=15)
    img2ts.calc()
