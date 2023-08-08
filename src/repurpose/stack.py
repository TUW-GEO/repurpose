import warnings
from pygeogrids.grids import genreg_grid, CellGrid, BasicGrid
import numpy as np
import xarray as xr
import pandas as pd

"""
TODOs: 
- maybe move image module to pynetcf at some point
- string variables are currently not supported
- maybe use x/y instead of lat/lon in the future.
    # Example - your x and y coordinates are in a Lambert Conformal projection
    data_crs = ccrs.LambertConformal(central_longitude=-100)
    # Transform the point - src_crs is always Plate Carree for lat/lon grid
    x, y = data_crs.transform_point(-122.68, 21.2, src_crs=ccrs.PlateCarree())
    # Now you can select data
    ds.sel(x=x, y=y)
"""


class Regular3dimImageStack:

    def __init__(self,
                 grid,
                 timestamps,
                 time_collocation=True,
                 reference_time=None,
                 zlib=True):
        """
        Combines a full (no gaps), regular grid with a xarray 3dim data set
        (time, lat, lon).
        A regular gridded image stack is a collection of images containing
        different variables for a set of dates that are stored in a single
        dataset object.
        Fits for data from e.g models (e.g. ERA5), but also some satellite
        products (ESA CCI).
        This data format is NOT suitable to store swath data / L2 products,
        observations on irregular grids (e.g. H SAF ASCAT) or in situ data.

        Parameters
        ----------
        grid : BasicGrid
            Must be a full (no gaps), regular, grid with the correct 2D shape
            assigned.
        timestamps: pd.DatetimeIndex
            The timestamps of the image stack. Must contain at least one time
             stamp and no duplicates.
        time_collocation: bool, optional (default: True)
            If True, then upon writing a time series into the stack
            (`write_ts`), we look for the closest timestamp in the stack and
            assign the data to this timestamp. The difference between the
            original timestamp and the time image stack time stamp is
            transferred into a new variable 'timedelta_seconds'. If there are
            more time stamps in a time series than in the image stack, they
            might be lost.
            If this is False, then time stamps passed via `write_ts` must match
            exactly to the time stamps in the stack.
        reference_time: str, optional (default: None)
            The reference time used when converting time stamps into numbers
            for storing them in files. If None is passed, then the first time
            stamp is used as the reference time.
        zlib: bool, optional (default: True)
            If True, chunk based compression with zlib (complevel 6) is
            applied for all numeric variables.
        """
        self.grid: CellGrid = self._eval_grid(grid.to_cell_grid(5.))

        if timestamps.has_duplicates:
            raise ValueError("Timestamps must not contain duplicates.")

        self.timestamps = timestamps.sort_values()
        self.reference_time = pd.to_datetime(reference_time) or \
                              self.timestamps[0].to_pydatetime()

        self.time_collocation = time_collocation
        self.timedelta_seconds = "timedelta_seconds"  # protected variable

        self.zlib = zlib

        self.ds = None
        self._init_dataset()

    @staticmethod
    def _eval_grid(img_grid) -> CellGrid:
        """
        Make sure that the grid works for the image writer
        """

        if not isinstance(img_grid, (CellGrid, BasicGrid)):
            raise TypeError("img_grid must be a pygeogrids Grid")

        if isinstance(img_grid, BasicGrid):
            img_grid = img_grid.to_cell_grid(5, 5)

        if not len(img_grid.shape) == 2:
            raise ValueError("img_grid shape must be 2D")
        if not (img_grid.shape[0] * img_grid.shape[1] == len(
                img_grid.activegpis)):
            raise ValueError(
                "Shape does not match to the number of points in the grid, "
                "image grid must not contain gaps be gap free")

        # check that gpis are sored from - to + latitudes (origin bottom)
        sort = np.argsort(img_grid.activegpis)
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        if not is_sorted(img_grid.activearrlat[sort]):
            raise ValueError("Grid origin must be in the bottom left corner, "
                             "i.e. GPIs are sorted from - to + latitudes")

        return img_grid

    @classmethod
    def from_genreg(cls, resolution=0.25, extent=None, **kwargs):
        """
        Initialize image stack from regular raster of the passed resolution.

        Parameters
        ----------
        resolution: float, optional (default: 0.25)
            Resolution in degrees.
            A global raster of the chosen resolution is created.
        extent: list or None
            Extent of the output image as [minlat, maxlat, minlon, maxlon].
        """
        grid = genreg_grid(
            resolution, resolution, *(extent or []), origin="bottom")
        return cls(grid, **kwargs)

    def _init_dataset(self):
        """
        Initialise a data stack for the passed bounding cell or globally.
        Data can then be added to the subset.
        """
        latdim = np.unique(self.grid.activearrlat)
        londim = np.unique(self.grid.activearrlon)

        self.ds = xr.Dataset(
            coords=dict(
                lon=londim,
                lat=latdim,
                time=self.timestamps,
                reference_time=self.reference_time))

        lon_attr = {
            "standard_name": "longitude",
            "long_name": "location longitude",
            "units": "degrees_east",
            "valid_range": (-180.0, 180.0)
        }
        self.ds['lon'].attrs.update(lon_attr)

        lat_attr = {
            "standard_name": "latitude",
            "long_name": "location latitude",
            "units": "degrees_north",
            "valid_range": (-90.0, 90.0)
        }
        self.ds['lat'].attrs.update(lat_attr)

        time_attr = {
            "standard_name": "time",
            "long_name": "time",
        }
        self.ds['time'].attrs.update(time_attr)

        if self.time_collocation:
            self.add_variable(
                self.timedelta_seconds,
                static=False,
                dtype='float32',
                attrs=dict(
                    units="Seconds",
                    description="Offset from file-wide timestamp"))

    def add_variable(self,
                     name,
                     values=np.nan,
                     static=False,
                     attrs=None,
                     dtype='float32'):
        """
        Add (empty) variable data to the current image stack.

        Parameters
        ----------
        name: str
            Name of the variable.
        values: float or np.ndarray, optional (default: np.nan)
            Value that the variable should be initialised with. If an array is
            passed, it must have the correct shape.
        static: bool, optional (default: False)
            If True, the variable is static, i.e. it has no time dimension.
        attrs: dict, optional (default: None)
            Attributes that should be assigned to the variable.
        dtype: str, optional (default: 'float32')
            Data type of the variable.
                'float32', 'float64'
                'int32', 'int64', 'int16', 'int8'
                'str' (not yet supported)
        """
        if name in self.ds:
            raise ValueError(f"Variable {name} already exists. Use the write"
                             f"functions to add data.")

        if not static and self.timestamps is None:
            raise ValueError("To add non-static variables, timestamps must be "
                             "passed.")
        if static:
            dims = ("lat", "lon")
            shape = self.grid.shape
        else:
            dims = ("time", "lat", "lon")
            shape = (len(self.timestamps), *self.grid.shape)

        if dtype == "str":
            is_string = True
            raise NotImplementedError("String support not yet implemented"
                                      f"in {self.__class__.__name__}")
        else:
            is_string = False

        if not isinstance(values, np.ndarray):
            values = np.full(shape, values, dtype=dtype)
        else:
            values = values.astype(dtype)

        self.ds[name] = xr.DataArray(
            dims=dims,
            data=values,
        )

        self.ds[name].attrs.update(attrs or {})

        if self.zlib and not is_string:
            self.ds[name].encoding.update(
                zlib=True,
                complevel=6,
            )

    @staticmethod
    def t_max_delta(dt):
        """
        Find max of deltas between passed time stamps.

        Parameters
        ----------
        dt: pd.DatetimeIndex
            Datetime index. Deltas are computed between subsequent values.

        Returns
        -------
        delta_h : float
            The max delta in hours between the passed time stamps
        """
        shift = np.roll(dt.values, -1)
        delta = shift[:-1] - dt[:-1]
        dmax = delta.max()
        d = dmax.days
        s = dmax.seconds

        return (d * 24) + (s / 3600)

    def collocate(self, df):
        """
        For each image time stamp find the closest time series time stamp
        afterwards. Then convert time series time stamps to deltas (>0) from
        the image time stamps.
        If the image stack sampling is too sparse, i.e. multiple time series
        time stamps are assigned to the same image, then some data might be
        lost.

        Parameters
        ----------
        df: pd.DataFrame
            Loaded time series data

        Returns
        -------
        collocated: pd.DataFrame
            Collocated version of df
        """
        window = self.t_max_delta(self.timestamps)
        window = pd.Timedelta(hours=window)

        reindex = pd.DataFrame(
            index=df.index, data={'__idx': np.arange(len(df))})

        reindex = reindex.reindex(
            self.timestamps, method="bfill", tolerance=window,
            limit=1).dropna()

        collocated = pd.DataFrame(
            index=reindex.index,
            columns=df.columns,
            data=df.iloc[reindex['__idx'], :].values)
        collocated['__index_other'] = df.iloc[reindex['__idx'].values].index

        collocated["__distance_other"] = (
            collocated["__index_other"] - collocated.index)

        # force int64, otherwise error on Windows
        offset_sec = collocated['__distance_other'].values.astype(
            np.int64) * 1e-9

        collocated[self.timedelta_seconds] = offset_sec
        collocated.drop(
            columns=['__index_other', '__distance_other'], inplace=True)

        return collocated

    def write_ts(self, df, gpi, new_var_kwargs=None):
        """
        Write time series for gpi to stack.

        Parameters
        ----------
        df: pd.DataFrame
            Data to be written to the stack. Columns contain variable names.
            If a variable is not yet present in the stack, a warning is issued.
        gpi: int
            Gpi of the grid cell where the data is written to.
        new_var_kwargs: dict[str, dict], optional (default: None)
            {variable_name: dict, ...}
            In case a variable is not yet in the data set, we use these kwargs
            (passed to add_variable) to add the variable to the data set.
            The key is the column in df, values are the kwargs passed to
            `add_variable`.
        """
        if df.index.has_duplicates:
            raise ValueError("Index must not contain duplicates.")

        df.sort_index(inplace=True)

        row, col = self.grid.gpi2rowcol(gpi)

        if self.time_collocation:
            df = self.collocate(df)
        else:
            comm = np.intersect1d(df.index.values, self.timestamps)
            df = df.loc[comm]

        new_var_kwargs = new_var_kwargs or {}

        for var in df.columns:
            if var not in self.ds.data_vars:
                if var in new_var_kwargs:
                    kwargs = new_var_kwargs[var]
                else:
                    kwargs = {}
                if ('static' in kwargs) and kwargs['static']:
                    warnings.warn("New variable must not be static.")
                    kwargs.pop('static')
                self.add_variable(var, **kwargs)

            if self.ds[var].ndim == 3:
                t = np.where(np.isin(self.timestamps, df.index))[0]
                self.ds[var].values[t, row, col] = df[var].values
            else:
                raise ValueError(f"Variable {var} is static.")

    def write_loc(self, gpis, data, timestamp=None, new_var_kwargs=None):
        """
        Write data for multiple gpis to one image.
        For static images (no time dimension), no timestamp is required.
        For dynamic images, a timestamp must be passed.

        Parameters
        ----------
        gpis: list of int
            List of gpis for which data is passed.
        data: dict[str, np.ndarray]
            Data to be written. Keys are variable names (must exist in the
            dataset). Shape of each array must be (len(gpis),).
        timestamp: str or datetime, optional (default: None)
            Timestamp of the image. If None, the image is static.
        new_var_kwargs: dict[str, dict], optional (default: None)
            {variable_name: dict, ...}
            In case a variable is not yet in the data set, we use these kwargs
            (passed to add_variable) to add the variable to the data set.
                        The key is the column in df, values are the kwargs passed to
            `add_variable`.
        """
        row, col = self.grid.gpi2rowcol(gpis)

        new_var_kwargs = new_var_kwargs or {}

        for var, dat in data.items():
            if var not in self.ds.data_vars:
                if var in new_var_kwargs:
                    kwargs = new_var_kwargs[var]
                else:
                    kwargs = {}
                kwargs['static'] = True if timestamp is None else False
                self.add_variable(var, **kwargs)
            if self.ds[var].ndim == 3:
                if timestamp is None:
                    raise ValueError(
                        "Timestamp must be passed for dynamic images.")
                if self.time_collocation and (timestamp
                                              not in self.timestamps):
                    # find the closest time stamp and add time deltas
                    delta = pd.to_datetime(timestamp) - self.timestamps
                    delta = delta.total_seconds().values
                    delta[delta < 0] = np.inf
                    t = np.argmin(delta)
                    dt = np.full((len(row),), delta[t])
                    self.write_loc(
                        gpis, {self.timedelta_seconds: dt},
                        timestamp=self.timestamps[t])
                else:
                    t = self.timestamps.get_loc(timestamp)
                self.ds[var].values[t, row, col] = dat
            else:
                self.ds[var].values[row, col] = dat

    def to_netcdf(self, path, *args, **kwargs):
        """
        Shortcut to xarray.Dataset.to_netcdf.
        Write current stack to file. Zlib compression is applied when selected
        for the data set for all numeric variables.
        Other compression options can be set via the encoding keyword, e.g.
           encoding={'sm': {'scale_factor': 0.001, 'dtype': 'int32',
                            '_FillValue': -9999}}
            to store sm values (0-1) with 3 decimal places precision as
            int32 (nans are stored as -9999)

        Parameters
        ----------
        path: str
            Path to output file.
        args, kwargs:
            Passed to `xarray.Dataset.to_netcdf()`.
        """
        if 'encoding' in kwargs:
            for name in kwargs['encoding']:
                if name in self.ds:
                    kwargs['encoding'][name].update(self.ds[name].encoding)

        self.ds.to_netcdf(path, *args, **kwargs)

    def close(self):
        self.ds.close()
