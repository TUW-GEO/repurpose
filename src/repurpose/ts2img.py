from repurpose.process import parallel_process_async, idx_chunks
import logging
import numpy as np
import pandas as pd
import xarray as xr
from pygeogrids.grids import BasicGrid, CellGrid
import os
from typing import Union
from pynetcf.time_series import (
    GriddedNcOrthoMultiTs,
    GriddedNcContiguousRaggedTs,
    GriddedNcIndexedRaggedTs
)
from datetime import datetime
import warnings

from repurpose.stack import Regular3dimImageStack

"""
TODOs:
- add possibility to use resampling methods other than nearest neighbour
    - integrate repurpose.resample module
    - allows weighting functions etc.
- similar to resample, use multiple neighbours when available for image pixel
- further harmonisation with pynetcf interface
- time ranges for images instead of time stamps
"""

def _convert(converter: 'Ts2Img',
             writer: Regular3dimImageStack,
             img_gpis: np.ndarray,
             lons: np.ndarray,
             lats: np.ndarray,
             preprocess_func=None,
             preprocess_kwargs=None) -> xr.Dataset:
    """
    Wrapper to allow parallelization of the conversion process.
    This is kept outside the Ts2Img class for parallelization.
    """
    for gpi, lon, lat in zip(img_gpis, lons, lats):
        ts = converter._read_nn(lon, lat)
        if ts is None:
            continue
        if preprocess_func is not None:
            preprocess_kwargs = preprocess_kwargs or {}
            ts = preprocess_func(ts, **preprocess_kwargs)
        if np.any(np.isin(ts.columns, Ts2Img._protected_vars)):
            raise ValueError(
                f"Time series contains protected variables. "
                f"Please rename them: {Ts2Img._protected_vars}"
            )
        writer.write_ts(ts, gpi)
    return writer.ds


def _write_img(
        image: xr.Dataset,
        dt: datetime,
        out_path: str,
        filename: str,
        annual_folder: bool = True,
        encoding: dict = None,
):
    """
    Wrapper to allow writing several images in parallel (see
    Ts2Img.store_netcdf_images).
    This is kept outside the Ts2Img class for parallelization.
    """
    if annual_folder:
        out_path = os.path.join(out_path, f"{dt.year:04}")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    image.attrs['date_created'] = f"File created: {datetime.now()}"
    image.to_netcdf(os.path.join(out_path, filename),
                    engine='netcdf4', encoding=encoding)

    image.close()


class Ts2Img:

    """
    Takes a time series dataset and converts it into a set of images.
    Images are stored on a regular grid. This includes a spatial and temporal
    lookup, ie resampling of the time series data to a regular 2d grid as well
    as assigning time series time stamps to images.

    Protected variable names (used internally) are:
        timedelta_seconds, index_other, distance_other
        gpi, lon, lat

    Parameters
    ----------
    ts_reader: GriddedNcOrthoMultiTs or GriddedNcContiguousRaggedTs or GriddedNcIndexedRaggedTs
        A reader that returns a time series for a given lon/lat combination.
        The class method defined in `read_name` is called to read a pandas
        DataFrame that has a DateTimeIndex and the variables as columns for
        a location.
    img_grid: BasicGrid or CellGrid
        A regular grid that defines the output images. Must be rectangular and
        have a 2d shape attribute. Can be a spatial subset of the time series
        grid and contain points that are missing in the time series (filled
        with nan). For each grid point, we search the closest time series
        (within `max_dist` of ts_reader).
    timestamps: pd.DateTimeIndex
        Each data point in the loaded time series must be assigned to an image.
        This defines the temporal sampling of the image stack. Each time stamp
        is a separate image.
        The closest time stamp from the time series will be stored in the
        according image, other data that would be assinged to the same image
        are DISCARDED!
        In this case a higher frequency (eg 12-hourly) should be chosen.
        A too low frequency here means that information is lost.
        A too high frequency here means that data is split up into many images.
    variables: dict or list[str] or None, optional (default: None)
        Data variables to be read from the time series and transfer to the
        images. Must exist in the time series. If a dict is given, then the
        variables are renamed after reading.
        Ideally a fill value for each variable (new name) is given in
        'fill_values'.
        If None, all variables are read.
    read_function: str, optional (default: 'read')
        Name of the method in `ts_reader` that takes a lon/lat pair and returns
        a pandas DataFrame with a DateTimeIndex and the variables as columns.
    max_dist: float, optional (default: 0.25)
        Maximum distance around an image grid cell to tool for a time series.
        If mutliple are found, only the nearest one is used!
    time_colloction: bool, optional (default: True)
        Relevant when converting data with varying time stamps per location.
        For each image time stamp find the closest time series time stamp
        afterwards. Then convert time series time stamps to deltas (>0) from
        the image time stamps and store them in a new image variable
        'timedelta_seconds'.
    loglevel: str, optional (default: 'WARNING')
        Logging level.
        Must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    """

    # Some variables are generated internally and cannot be used.
    # If there are variable with the same name, rename them using the
    # `rename` keyword of the `.calc()` method.

    _protected_vars = ['timedelta_seconds']

    def __init__(self, ts_reader, img_grid, timestamps,
                 variables=None, read_function='read',
                 max_dist=18000, time_collocation=True, loglevel="WARNING"):

        self.ts_reader = ts_reader
        self.img_grid: CellGrid = Regular3dimImageStack._eval_grid(img_grid)
        self.timestamps = timestamps

        if variables is not None:
            if not isinstance(variables, dict):
                variables = {v: v for v in variables}
        self.variables = variables

        self.read_function = read_function
        self.max_dist = max_dist
        self.time_collocation = time_collocation

        self.loglevel = loglevel
        self.stack = None


    def _cell_writer(self, cell: int, timestamps_chunk: pd.DatetimeIndex):
        """
        Create sub-cube for cell to write time series of current chunk to.
        This is only in memory, writing to disk is done after collecting all
        cells.
        """
        # RegularGriddedImageStack for a single cell
        cellgrid = self.img_grid.subgrid_from_cells(cell)
        n_lats = np.unique(cellgrid.activearrlat).size
        n_lons = np.unique(cellgrid.activearrlon).size
        cellgrid.shape = (n_lats, n_lons)

        writer = Regular3dimImageStack(
            cellgrid, timestamps=timestamps_chunk, zlib=False,
            time_collocation=self.time_collocation)

        return writer

    def _read_nn(self, lon: float, lat: float) -> Union[pd.DataFrame, None]:
        """
        Wrapper around read function. Read nearest time series within max_dist.
        Log error when no GPI is found in time series + rename columns if
        necessary.
        """
        _, dist = self.ts_reader.grid.find_nearest_gpi(lon, lat)
        if dist > self.max_dist:
            return None
        try:
            ts = self.ts_reader.__getattribute__(self.read_function)(lon, lat)
        except Exception as e:
            logging.error(f"Error reading Time series data at "
                          f"lon: {lon}, lat: {lat}: {e}")
            return None
        if (self.variables is not None) and (ts is not None):
            ts = ts.rename(columns=self.variables)[self.variables.values()]
        return ts

    def _calc_chunk(self, timestamps, preprocess_func=None, preprocess_kwargs=None,
                    log_path=None, n_proc=1):
        """
        Create image stack from time series for the passed timestamps.
        See: self.calc
        """
        self.timestamps = timestamps
        logging.info(f"Processing chunk from {timestamps[0]} to "
                     f"{timestamps[-1]}")

        # Transfer time series to images, parallel for cells
        STATIC_KWARGS = {
            'converter': self,
            'preprocess_func': preprocess_func,
            'preprocess_kwargs': preprocess_kwargs,
        }
        ITER_KWARGS = {'writer': [], 'img_gpis': [], 'lons': [], 'lats': []}

        for cell in np.unique(self.img_grid.activearrcell):
            ITER_KWARGS['writer'].append(self._cell_writer(cell, timestamps))
            img_gpi, lons, lats = self.img_grid.grid_points_for_cell(cell)
            ITER_KWARGS['img_gpis'].append(img_gpi)
            ITER_KWARGS['lons'].append(lons)
            ITER_KWARGS['lats'].append(lats)

        stack = parallel_process_async(
            _convert, ITER_KWARGS, STATIC_KWARGS, n_proc=n_proc,
            show_progress_bars=True, log_path=log_path,
            verbose=False, ignore_errors=True)

        stack = xr.combine_by_coords(stack)

        # positive latitudes are on top
        stack = stack.reindex(dict(time=stack['time'],
                                   lat=stack['lat'][::-1],
                                   lon=stack['lon']))
        return stack

    def calc(self, path_out, format_out='slice', preprocess=None,
             preprocess_kwargs=None, postprocess=None, postprocess_kwargs=None,
             fn_template="{datetime}.nc",
             drop_empty=False, encoding=None, zlib=True, glob_attrs=None,
             var_attrs=None, var_fillvalues=None, var_dtypes=None,
             img_buffer=100, n_proc=1):
        """
        Perform conversion of all time series to images. This will first split
        timestamps into processing chunks (img_buffer) and then - for each
        chunk - iterate through all cells (parallel) in the img_grid, and
        transfer the time series for each pixel into the image stack.

        Parameters
        ----------
        path_out: str
            Path to the output directory where the files are written to.
        format_out: str, optional (default: 'slice')
            - slice: write each time step as a separate file. In this case
                the fn_template must contain a placeholder {datetime} where
                the date is inserted for each image
            - stack: write all time steps into one file. In this case if there
                is a {datetime} placeholder in the fn_template, then the time
                range is inserted.
        preprocess: callable, optional (default: None)
            Function that is applied to each time series before converting it.
            The first argument is the data frame that the reader returns.
            Additional keyword arguments can be passed via `preprocess_kwargs`.
            The function must return a data frame of the same form as the input
            data, i.e. with a datetime index and at least one column of data.
            Note: As an alternative to a preprocessing function, consider
            applying an adapter to the reader class. Adapters also perform
            preprocessing, see `pytesmo.validation_framework.adapters`
            A simple example for a preprocessing function to compute the sum:
            ```
            def preprocess_add(df: pd.DataFrame, **preprocess_kwargs) \
                    -> pd.DataFrame:
                df['var3'] = df['var1'] + df['var2']
                return df
            ```
        preprocess_kwargs: dict, optional (default: None)
            Keyword arguments for the preprocess function. If None are given,
            then the preprocessing function is is called with only the input
            data frame and no additional arguments (see example above).
        postprocess: Callable, optional (default: None)
            Function that is applied to the image stack after loading the data
            and before writing it to disk. The function must take xarray
            Dataset as the first argument and return an xarray Dataset of the
            same form as the input data.
            A simple example for a preprocessing function to add a new variable
            from the sum of two existing variables:
            ```
            def preprocess_add(stack: xr.Dataset, **postprocess_kwargs) \
                    -> xr.Dataset
                stack = stack.assign(var3=lambda x: x['var0'] + x['var2'])
                return stack
            ```
        postprocess_kwargs: dict, optional (default: None)
            Keyword arguments for the postprocess function. If None are given,
            then the postprocess function is called with only the input
            image stack and no additional arguments (see example above).
        fn_template: str, optional (default: "{datetime}.nc")
            Template for the output image file names.
            If format_out is 'slice', then a placeholder {datetime} must be
            in the fn_template, which will be replaced by the timestamp of each
            image.
            If format_out is 'stack', then no {datetime} placeholder is
            required. If it's till provided, the time range of the stack is
            inserted.
        drop_empty: bool, optional (default: False)
            Images where all data variables are empty are removed from
            the stack after loading / before writing. Otherwise, emtpy images
            will be written to disk.
        encoding: dict of dicts, optional (default: None)
            Encoding kwargs for each variable. Are passed to netcdf for storing
            the files to apply dtype, scale_factor, add_offset, etc.
            Make sure that the encoding is consistent with the data and fill
            values (`var_fillvalues`).
            For example, conversion to int16 for data values between 0 and 100
            can result in data loss.
            e.g. {'sm': {'dtype': 'int32', 'scale_factor': 0.01}}
        zlib: bool, optional (default: True)
            If True, then the netcdf files are compressed using zlib
            compression for all data variables.
        glob_attrs: dict, optional (default: None)
            Additional global attributes that are added to the netcdf file.
            e.g. {'product': 'ASCAT 12.5 TS'}
        var_attrs: dict of dicts, optional (default: None)
            Additional variable attributes that are added to the netcdf file.
            The dict must have the following structure:
            {varname: {'attrname': value}}, e.g
            {'sm': {'long_name': 'soil moisture', 'units': 'm3 m-3'}, ...}
            In case variable was renamed, use the new name here!
        var_fillvalues: dict, optional (default: None)
            Fill values for each variable. By default, nan is used for all
            variables (you can also use the `encoding` parameter to set a
            fill value when writing to disk).
            In case variable was renamed, use the new name here!
        var_dtypes: dict, optional (default: None)
            Data types for each variable. By default, float32 is used for all
            variables (you can also use the `encoding` parameter to set a
            dtype when writing to disk).
            In case variable was renamed, use the new name here!
        img_buffer: int, optional (default: 100)
            Size of the stack before writing to disk. Larger stacks need
            more memory but will lead to faster conversion.
            Passing -1 means that the whole stack loaded into memory at once.
        n_proc: int, optional (default: 1)
            Number of processes to use for parallel processing. We parallelize
            by 5 deg. grid cell.
        """

        if format_out not in ['slice', 'stack']:
            raise ValueError("format_out must be 'slice' or 'stack'")
        if format_out == 'slice' and '{datetime}' not in fn_template:
            raise ValueError("fn_template must contain {datetime} for "
                             "format_out='slice'")

        log_path = os.path.join(path_out, '000_logs')
        os.makedirs(log_path, exist_ok=True)

        dt_index_chunks = list(idx_chunks(self.timestamps, int(img_buffer)))

        for timestamps in dt_index_chunks:
            self.stack = self._calc_chunk(timestamps,
                                          preprocess, preprocess_kwargs,
                                          log_path, n_proc)

            if drop_empty:
                vars = [var for var in self.stack.data_vars if var not in
                        self._protected_vars]
                idx_empty = []
                for i in range(len(self.stack.time)):
                    img = self.stack[vars].isel(time=i)
                    for var in vars:
                        if not np.all([np.all(np.isnan(img[var].values))]):
                            break
                    else:
                        idx_empty.append(i)

                self.stack = self.stack.drop_isel(time=idx_empty)

            if postprocess is not None:
                postprocess_kwargs = postprocess_kwargs or {}
                self.stack = postprocess(self.stack, **postprocess_kwargs)

            if var_fillvalues is not None:
                for var, fillvalue in var_fillvalues.items():
                    self.stack[var].values = np.nan_to_num(
                        self.stack[var].values, nan=fillvalue)

            if var_dtypes is not None:
                for var, dtype in var_dtypes.items():
                    self.stack[var] = self.stack[var].astype(dtype)

            encoding = encoding or {}

            # activate zlib compression for all data variables
            if zlib is True:
                for var in self.stack.data_vars:
                    if var not in encoding:
                        encoding[var] = {}
                    encoding[var]['zlib'] = True
                    encoding[var]['complevel'] = 6

            if glob_attrs is not None:
                self.stack.attrs.update(glob_attrs)
            if var_attrs is not None:
                for var in var_attrs:
                    self.stack[var].attrs.update(var_attrs[var])

            if self.stack['time'].size == 0:
                warnings.warn("No images in stack to write to disk.")
                self.stack = None
            elif format_out == 'stack':
                if '{datetime}' in fn_template:
                    dt_from = pd.to_datetime(self.stack.time.values[0])\
                        .to_pydatetime().strftime('%Y%m%dT%H%M%S')
                    dt_to = pd.to_datetime(self.stack.time.values[-1])\
                        .to_pydatetime().strftime('%Y%m%dT%H%M%S')
                    fname = fn_template.format(datetime=f"{dt_from}_{dt_to}")
                else:
                    fname = fn_template
                self.stack.to_netcdf(os.path.join(path_out, fname),
                                     encoding=encoding)
                self.stack = None
            elif format_out == 'slice':
                self.store_netcdf_images(
                    path_out, fn_template, keep=False, encoding=encoding,
                    annual_folder=True, n_proc=n_proc)
            else:
                raise NotImplementedError("Unknown `format_out`")

    def store_netcdf_images(self, path_out, fn_template=f"{datetime}.nc",
                            encoding=None, annual_folder=True,
                            keep=False, n_proc=1):
        """
        Write the (global) merged image stack to netcdf files.

        Parameters
        ----------
        path_out: str
            Path to the output directory where the files are written to.
        fn_template: str, optional (default: None)
            Template for the output image file names. Must contain a placeholder
            {datetime} where the image date is inserted.
        encoding: dict (default: None)
            Encoding for the netcdf variables. The keys are the variable names,
            If True, then the images are grouped by year, and images for each
            year a written to a separate folder.
        annual_folder: bool, optional (default: True)
            If True, then the images are grouped by year, and images for each
            year a written to a separate folder.
        keep: bool, optional (default: False)
            If True, then the image stack is kept in memory during writing.
            This is only needed if anything else should be done with the stack
            after writing it to disk.
            If False (recommended), then the stack is gradually deleted to empty
            memory during writing.
        n_proc: int, optional (default: 1)
            Number of processes to use for reading cells and writing images
            in parallel. Merging cells after reading is not parallelised and
            might be a bottleneck.
        """
        # Slice stack and write down individual images with according file
        # names
        if self.stack is None:
            warnings.warn("No stack loaded, nothing to write")

        images = []
        dts = []
        filenames = []
        drop_z = []
        for z in self.stack['time'].values:
            dt = pd.to_datetime(z).to_pydatetime()
            filename = fn_template.format(
                datetime=datetime.strftime(dt, '%Y%m%d%H%M%S'))
            image = self.stack.sel({'time': [dt]})

            image.attrs['id'] = filename

            drop_z.append(dt)
            images.append(image)
            dts.append(dt)
            filenames.append(filename)

            if not keep:
                if len(drop_z) > 10:  # to avoid too many duplicates in memory
                    self.stack = self.stack.drop_sel({'time': drop_z})
                    drop_z = []

        if not keep:
            del self.stack
            self.stack = None

        _ = parallel_process_async(
            FUNC=_write_img,
            ITER_KWARGS={'image': images, 'filename': filenames,
                         'dt': dts},
            STATIC_KWARGS={'out_path': path_out,
                           'annual_folder': annual_folder,
                           'encoding': encoding},
            n_proc=n_proc,
            show_progress_bars=True,
            verbose=False,
            loglevel=self.loglevel,
            ignore_errors=True,
        )
