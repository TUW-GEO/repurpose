import warnings
import platform
from repurpose.process import parallel_process_async, idx_chunks
import pynetcf.time_series as nc
from pygeogrids.grids import CellGrid
import pygeogrids.grids as grids
import repurpose.resample as resamp
import numpy as np
import os
from datetime import datetime
import logging
import pygeogrids.netcdf as grid2nc
import pandas as pd
from pygeobase.object_base import Image
import sharedmem as mem


class Img2TsError(Exception):
    pass


class Img2Ts:
    """
    class that uses the read_img iterator of the input_data dataset
    to read all images between startdate and enddate and saves them
    in netCDF time series files according to the given netCDF class
    and the cell structure of the outputgrid

    Parameters
    ----------
    input_dataset : DatasetImgBase like class instance or ImageBaseConnection
        must implement a ``read(date, **input_kwargs)`` iterator that returns a
        `pygeobase.object_base.Image`.
    outputpath : str
        path where to save the time series to
    startdate : datetime.datetime
        date from which the time series should start. Of course images
        have to be available from this date onwards.
    enddate : datetime.datetime
        date when the time series should end. Images should be available
        up until this date
    input_kwargs : dict, optional (default: None)
        keyword arguments which should be used in the read_img method of the
        input_dataset
    input_grid : CellGrid, optional
        the grid on which input data is stored.
        If not given then the grid of the input dataset will be used.
        If the input dataset has no grid object then resampling to the
        target_grid is performed.
    target_grid : CellGrid, optional
        the grid on which the time series will be stored.
        If not given then the grid of the input dataset will be used
    imgbuffer : int, optional
        number of days worth of images that should be read into memory before
        a time series is written. This parameter should be chosen so that
        the memory of your machine is utilized. It depends on the daily data
        volume of the input dataset. If -1 is passed, all available
        data will be loaded at once (no buffer).
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
        resample methods to use if resampling is necessary, either 'nn' for
        nearest neighbour or 'custom' for custom weight function.
        Can also be a dictionary in which the method is specified for each
        variable
    r_weightf : function or dict, optional
        if r_methods is custom this function will be used to calculate the
        weights depending on distance. This can also be a dict with a separate
        weight function for each variable.
    r_min_n : int, optional
        Minimum number of neighbours on the target_grid that are required for
        a point to be resampled.
    r_radius : float, optional
        resample radius in which neighbours should be searched given in meters
    r_neigh : int, optional
        maximum number of neighbours found inside r_radius to use during
        resampling. If more are found the r_neigh closest neighbours will be
        used.
    r_fill_values : number or dict, optional
        if given the resampled output array will be filled with this value if
        no valid resampled value could be computed, if not a masked array will
        be returned can also be a dict with a fill value for each variable
    filename_templ : string, optional
        filename template must be a string with a string formatter for the
        cell number.
        e.g. '%04d.nc' will translate to the filename '0001.nc' for cell
        number 1.
    gridname : string, optional
        filename of the grid which will be saved as netCDF
    global_attr : dict, optional
        global attributes for each file
    ts_attributes : dict, optional
        dictionary of attributes that should be set for the netCDF time series.
        Can be either a dictionary of attributes that will be set for all
        variables in input_data or a dictionary of dictionaries.
        In the second case the first dictionary has to have a key for each
        variable returned by input_data and the second level dictionary will be
        the dictionary of attributes for this time series.
    ts_dtype : numpy.dtype or dict of numpy.dtypes
        data type to use for the time series, if it is a dict then a key must
        exist for each variable returned by input_data.
        Default : None, no change from input data
    time_units : string, optional
        units the time axis is given in.
        Default: "days since  1858-11-17 00:00:00" which is modified julian
        date for regular images this can be set freely since the conversion
        is done automatically, for images with irregular timestamp this will
        be ignored for now
    zlib: boolean, optional (default: True)
        if True the saved netCDF files will be compressed
        Default: True
    n_proc: int, optional (default: 1)
        Number of parallel processes. Multiprocessing is only used when
        `n_proc` > 1. Applies to data reading and writing.
    """

    def __init__(self,
                 input_dataset, outputpath, startdate, enddate,
                 input_kwargs=None, input_grid=None, target_grid=None,
                 imgbuffer=100, variable_rename=None, unlim_chunksize=100,
                 cellsize_lat=180.0, cellsize_lon=360.0,
                 r_methods='nn', r_weightf=None, r_min_n=1, r_radius=18000,
                 r_neigh=8, r_fill_values=None, filename_templ='%04d.nc',
                 gridname='grid.nc', global_attr=None, ts_attributes=None,
                 ts_dtypes=None, time_units="days since 1858-11-17 00:00:00",
                 zlib=True, n_proc=1):

        self.imgin = input_dataset
        self.zlib = zlib
        if not hasattr(self.imgin, 'grid'):
            self.input_grid = input_grid
        else:
            self.input_grid = self.imgin.glob_grid

        if self.input_grid is None and target_grid is None:
            raise ValueError("Either the input dataset has to have a grid, "
                             "input_grid has to be specified or "
                             "target_grid has to be set")

        self.input_kwargs = input_kwargs or dict()

        self.target_grid = target_grid
        if self.target_grid is None:
            self.target_grid = self.input_grid
            self.resample = False
        else:
            # if input and target grid are not equal resampling is required
            if self.input_grid != self.target_grid:
                self.resample = True

        # if the target grid is not a cell grid make it one
        # default is just one cell for the entire grid
        if not isinstance(self.target_grid, grids.CellGrid):
            self.target_grid = self.target_grid.to_cell_grid(
                cellsize_lat=cellsize_lat, cellsize_lon=cellsize_lon)

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

        self.filename_templ = filename_templ
        self.global_attr = global_attr
        self.ts_attributes = ts_attributes
        self.ts_dtypes = ts_dtypes
        self.time_units = time_units
        self.non_ortho_time_units = "days since  1858-11-17 00:00:00"

        # if each image has only one timestamp then we are handling
        # time series of type Orthogonal multidimensional array representation
        # according to the CF conventions.
        # The following are detected from data and set during reading
        self.orthogonal = None  # to be set when reading data
        self.timekey = None  # to be set when reading data

        # Multiprocessing only used when n_proc > 1 chosen
        if platform.system().lower() != "linux" and n_proc != 1:
            warnings.warn("Parallel processing is for now only supported "
                          "on Linux systems. Setting `n_proc=1`.")
            self.n_proc = 1
        else:
            self.n_proc = n_proc

    def _read_image(self, date, target_grid):
        """
        Function to parallelize reading image data from dataset.
        Do not modify any class attributes here!

        Parameters
        ----------
        date: datetime.datetime
            Time stamp of the image to read
        target_grid: CellGrid, optional (default: None)
            To perform spatial resampling, a target grid is needed. If None is
            given, then no spatial resampling is performed.
            Only used if `resample=True`.

        Returns
        -------
        image: Image
            Image data read from input data set (might be spatially resampled)
        orthogonal: bool
            Whether the image fits the orthogonal time series format or not.
        """

        # optional on-the-fly spatial resampling
        resample_kwargs = {
            'methods': self.r_methods,
            'weight_funcs': self.r_weightf,
            'min_neighbours': self.r_min_n,
            'search_rad': self.r_radius,
            'neighbours': self.r_neigh,
            'fill_values': self.r_fill_values,
        }

        try:
            image = self.imgin.read(date, **self.input_kwargs)
        except IOError as e:
            logging.error("I/O error({0}): {1}".format(e.errno, e.strerror))
            return None

        logging.info(f"Read image with constant time stamp. "
                     f"Timestamp: {image.timestamp.isoformat()}")

        if self.resample:
            if target_grid is None:
                raise Img2TsError("Target grid is required for spatial "
                                  "resampling.")

            data = resamp.resample_to_grid(
                image.data, image.lon, image.lat,
                target_grid.activearrlon,
                target_grid.activearrlat,
                **resample_kwargs)

            # new image instance with resampled data
            metadata = image.metadata
            metadata["resampling_date"] = f"{datetime.now()}"
            image = Image(target_grid.activearrlon,
                          target_grid.activearrlat,
                          data=data,
                          metadata=metadata,
                          timestamp=image.timestamp,
                          timekey=image.timekey,
                          )

        orthogonal = self.orthogonal

        if image.timekey is not None:
            # if time_arr is not None means that each observation of the
            # image has its own observation time
            # this means that the resulting time series is not
            # regularly spaced in time
            if orthogonal is None:
                orthogonal = False
            else:
                if orthogonal:
                    raise Img2TsError(
                        "Images can not switch between a fixed image "
                        "timestamp and individual timestamps for each "
                        "observation")
        else:
            if orthogonal is None:
                orthogonal = True
            else:
                if not orthogonal:
                    raise Img2TsError(
                        "Images can not switch between a fixed image "
                        "timestamp and individual timestamps for each "
                        "observation")

        return image, orthogonal

    def _write_orthogonal(self,
                          cell: int,
                          target_grid: CellGrid,
                          celldata: dict,
                          timestamps: np.ndarray):
        """
        Write time series in OrthoMultiTs format.

        Parameters
        ----------
        cell: int
            Cell number of the data to write
        target_grid: CellGrid
            Grid containing time series localtions.
        celldata : dict
            dictionary with variable names as keys and 2D numpy.arrays as
            values
        timestamps: numpy.ndarray
            Array of datetime objects with same size as second dimension of
            data arrays.
        """
        cell_gpis, cell_lons, cell_lats = \
            target_grid.grid_points_for_cell(cell)

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

            dataout.write_all(cell_gpis, celldata, timestamps,
                              lons=cell_lons,
                              lats=cell_lats,
                              attributes=self.ts_attributes)

    def _write_non_orthogonal(self,
                              cell: int,
                              target_grid: CellGrid,
                              celldata: dict):
        """
        Write time series data for cell to IndexedRagged format.

        Parameters
        ----------
        cell: int
            Cell number
        target_grid: CellGrid
            Cell grid that the cell number refers to
        celldata: dict[str, np.ndarray]
            Time series data for netcdf data variables.
            arrays have the following shape [dates, ...]
        cell_index: np.ndarray
            Inidces of cell points in the global stack
        """
        cell_gpis, cell_lons, cell_lats = \
            target_grid.grid_points_for_cell(cell)

        fname = os.path.join(self.outputpath, self.filename_templ % cell)

        with nc.IndexedRaggedTs(
                fname,
                n_loc=cell_gpis.size,
                mode='a',
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
            for i, (gpi, gpi_lon, gpi_lat) in enumerate(
                    zip(cell_gpis, cell_lons, cell_lats)):
                gpi_data = {}
                # convert to modified julian date
                gpi_jd = celldata[self.timekey][i, :] - 2400000.5
                # remove measurements that were filled with the fill value
                # during resampling
                # doing this on the basis of the time variable should
                # be enough since without time -> no valid
                # observations
                if self.resample:
                    if self.r_fill_values is not None:
                        if type(self.r_fill_values) == dict:
                            time_fill_value = self.r_fill_values[self.timekey]
                        else:
                            time_fill_value = self.r_fill_values

                        valid_mask = gpi_jd != time_fill_value
                    else:
                        valid_mask = np.invert(gpi_jd.mask)
                    gpi_jd = gpi_jd[valid_mask]
                else:
                    # all are valid if no resampling took place
                    valid_mask = slice(None, None, None)

                if gpi_jd.size > 0:
                    for key in celldata:
                        if key == self.timekey:
                            continue
                        gpi_data[key] = celldata[key][i, valid_mask]

                    # transform into data frame
                    dataout.write(gpi, gpi_data, gpi_jd,
                                  lon=gpi_lon, lat=gpi_lat,
                                  attributes=self.ts_attributes,
                                  dates_direct=True)

    def _calc_cell(self, cell, img_stack_dict, timestamps, target_grid):
        """
        Select time series cell data from global stack and write to netcdf
        files.
        
        Parameters
        ----------
        cell: int
            Cell number in the target grid
        img_stack_dict: dict[str, mem.anonymousmemmap]
            Dict containing the global image stacks to convert. Shared
            between processes.
        timestamps: numpy.ndarray
            Array of datetime objects with same size as second dimension of
            data arrays.
        target_grid: CellGrid
            Target points for resampling and storing the time series on.
        """""
        # look where in the subset the data is
        cell_index = np.where(
            cell == target_grid.activearrcell)[0]

        if cell_index.size == 0:
            raise Img2TsError('cell not found in grid subset')

        celldata = {}

        for key in img_stack_dict.keys():
            # rename variable in output dataset
            if self.variable_rename is None:
                var_new_name = str(key)
            else:
                var_new_name = self.variable_rename[key]

            data = np.swapaxes(
                img_stack_dict.get(key)[:, cell_index], 0, 1)

            # change dtypes of output time series
            if self.ts_dtypes is not None:
                if type(self.ts_dtypes) == dict:
                    output_dtype = self.ts_dtypes[key]
                else:
                    output_dtype = self.ts_dtypes
                data = data.astype(output_dtype)

            celldata[var_new_name] = data

        if self.orthogonal:
            self._write_orthogonal(cell, target_grid, celldata, timestamps)
        elif not self.orthogonal:
            # time information is contained in `celldata`
            self._write_non_orthogonal(cell, target_grid, celldata)

    def calc(self):
        """
        Iterate through all images of the image stack and extract temporal
        chunks. Transpose the data and append it to the output time series
        files.
        """
        # save grid information in file
        grid2nc.save_grid(
            os.path.join(self.outputpath, self.gridname), self.target_grid)

        for img_stack_dict, timestamps in self.img_bulk():
            # ==================================================================
            start_time = datetime.now()
            cells = self.target_grid.get_cells()

            # temporally drop grids, due to issue when pickling them...
            target_grid = self.target_grid
            input_grid = self.input_grid
            self.target_grid = None
            self.input_grid = None

            from numpy.ctypeslib import as_ctypes

            if self.n_proc > 1:
                # shared image stack between parallel processes
                for k, v in img_stack_dict.items():
                    img_stack_dict[k] = mem.full_like(v, v)

            parallel_process_async(
                self._calc_cell,
                ITER_KWARGS={'cell': cells},
                STATIC_KWARGS={
                    'img_stack_dict': img_stack_dict,
                    'timestamps': timestamps,
                    'target_grid': target_grid,
                },
                log_path=os.path.join(self.outputpath, '000_log'),
                loglevel="INFO",
                n_proc=self.n_proc,
                show_progress_bars=False,
            )
            self.target_grid = target_grid
            self.input_grid = input_grid

            logging.info(f"Chunk processed in "
                         f"{datetime.now() - start_time} Seconds")

    def img_bulk(self):
        """
        Yields numpy array of images from imgbuffer between start and enddate
        until all images have been read.

        Returns
        -------
        img_stack_dict : dict[str, np.ndarray]
            stack of daily images for each variable
        startdate : datetime.datetime
            date of first image in stack
        enddate : datetime.datetime
            date of last image in stack
        datetimestack : np.ndarray
            array of the timestamps of each image
        jd_stack : np.ndarray or None
            None if all observations in an image have the same
            observation timestamp. Otherwise it gives the julian date
            of each observation in img_stack_dict

        Yields
        ------
        tuple[dict, datetime, np.ndarray or None]
        """

        timestamps = self.imgin.tstamps_for_daterange(
            self.startdate, self.enddate)

        for i, dates in enumerate(idx_chunks(pd.DatetimeIndex(timestamps),
                                             self.imgbuffer)):

            # Need to temporarily remove grids as they cannot be pickled for MP
            target_grid = self.target_grid
            input_grid = self.input_grid
            self.target_grid = None
            self.input_grid = None

            ITER_KWARGS = {'date': dates}

            results = parallel_process_async(
                self._read_image,
                ITER_KWARGS=ITER_KWARGS,
                STATIC_KWARGS={'target_grid': target_grid},
                show_progress_bars=False,
                log_path=os.path.join(self.outputpath, '000_log'),
                loglevel="INFO",
                n_proc=self.n_proc,
            )

            img_dict = {}
            timestamps = np.array([])

            while len(results) > 0:
                img, orthogonal = results.pop(0)

                for k, v in img.data.items():
                    if k not in img_dict:
                        img_dict[k] = []
                    img_dict[k].append(v)

                if self.orthogonal is None:
                    self.orthogonal = orthogonal
                if self.timekey is None:
                    self.timekey = img.timekey

                timestamps = np.append(timestamps, img.timestamp)

            order = np.argsort(timestamps)
            timestamps = timestamps[order]
            img_dict = {k: np.vstack(v)[order] for k, v in img_dict.items()}

            # Add the previous removed grids back
            self.target_grid = target_grid
            self.input_grid = input_grid

            yield (img_dict, timestamps)
