import numpy as np
import pandas as pd
from pygeogrids.grids import genreg_grid, BasicGrid, CellGrid

class Ts2Img(object):

    """
    Takes a time series dataset and converts it into a set of images.
    Images are stored on a regular grid.

    Parameters
    ----------
    tsreader: object
        Reader that has a 'read' method that returns a time series for a
        given lon/lat combination.
    variables: list[str]
        Data variables to be read from the time series and transfer to the
        images. Must exist in the time series. Ideally a fill value for each
        variable is given in 'fill_values'.
    startdate: str
        Start date of the image conversion.
    enddate: str
        End date of the image conversion.
    freq: str, optional (default: 'D')
        Frequency of the output images. Must be a valid pandas frequency.
    read_name: str, optional (default: 'read')
        Name of the method to read the time series.
    image_resolution: float, optional (default: 0.25)
        Resolution of the output images in degrees
    fn_template: str, optional (default: None)
        Template for the output image file names. Must contain a placeholder
        {datetime} where the image date is inserted.
    freq: str, optional (default: 'D')
        Frequency of the output images. Must be a valid pandas frequency.
    extent: list or None, optional (default: None)
        Extent of the output image as [minlat, maxlat, minlon, maxlon].
    fill_values: dict or None, optional (default: None)
        Fill values for each variable. The keys are the variable names, the
        values are the fill values. If a variable has no fill value, then
        np.nan is used. This might be inefficient and lead to e.g. integer
        variables being converted to float.
    ignore_time: bool, optional (default: False)
        If True, then the time stamps are ignored, and only the date of
        each time stamp is used. If False, then the time stamps are transferred
        into a new variable called 't0'. These are the offsets from the date
        in seconds.
    global_attrs: dict (default: None)
        Global attributes to be written to the output netcdf file.
    var_attrs: dict of dicts (default: None)
        Variable attributes to be written to the output netcdf file for each
        variable. The keys are the variable names, the values are dicts with
        the attribute names and values.
    img_buffer: int, optional (default: 100)
        Size of the stack before writing to disk.
    """

    def __init__(self, tsreader, variables, startdate, enddate, freq='D',
                 read_name='read', resolution=0.25, extent=None,
                 fill_values=None, ignore_time=False, fn_template="{date}.nc",
                 max_dist=np.inf, img_buffer=100):

        self.tsreader = tsreader
        self.read_name = read_name

        self.variables = variables

        self.startdate = startdate
        self.enddate = enddate
        self.freq = freq

        self.max_dist = max_dist
        self.ignore_time = ignore_time
        self.fn_templ = fn_template
        self.img_buffer = img_buffer

        self.out_grid = self._init_out_grid(resolution, extent)
        self.fill_values = self._init_fill_values(fill_values)

    def _init_fill_values(self, fill_values: dict or None) -> dict:
        # fill values for each variable
        if fill_values is None:
            fill_values = {}
        for var in self.variables:
            if var not in fill_values:
                fill_values[var] = np.nan
        return fill_values

    def _init_out_grid(self, res: float, ext: list or None) -> CellGrid:
        # target grid
        ext = ext if ext is not None else []
        return genreg_grid(res, *ext).to_cell_grid(5)

    def timeoffset(self, index, ds):
        """
        Time offset between the
        Parameters
        ----------
        ds

        Returns
        -------

        """

    def _read_chunk(self, cell):
        """
        Read all time series for a given cell. Create sub-cube as xarray
        Dataset.
        """
        grid_cell = self.out_grid.subgrid_from_cells(cell)
        celldata = []
        index = pd.date_range(self.startdate, self.enddate, freq=self.freq)
        for i, (_, lon, lat, _) in enumerate(grid_cell.get_grid_points()):
            gpi, dist = self.tsreader.grid.find_nearest_gpi(
                lon, lat, max_dist=self.max_dist)

            if (not isinstance(gpi, int)) or (dist > self.max_dist):
                ts = pd.DataFrame(index=index, data=self.fill_values)
            else:
                ts = getattr(self.tsreader, self.read_name)(lon, lat)

            ts.index.name = '__index'
            ts.reset_index(inplace=True)
            ts['TimeOffsetSeconds'] = ts['__index'].dt.time.apply(
                lambda x: x.hour * 3600 + x.minute * 60 + x.second)
            ts.set_index(ts['__index'], inplace=True)
            ts.set_index(ts.index.date, inplace=True)
            ts.drop(columns='__index', inplace=True)


            celldata.append(ts)

    #
    #
    # def convert(self):
    #     #create the output raster of the given resolution (cells)
    #
    #     # for each point, read the time series
    #     # combine
    #
    #     read all time series
    #         create a xr.Dataset for the cell
    #         read data for cell as xr Dataset
    #
    #
    # def calc(self, **tsaggkw):
    #     """
    #     does the conversion from time series to images
    #     """
    #     for gpis, ts in self.tsbulk(**tsaggkw):
    #         self.imgwriter.write_ts(gpis, ts)
    #
    # def tsbulk(self, gpis=None, **tsaggkw):
    #     """
    #     iterator over gpi and time series arrays of size self.ts_buffer
    #
    #     Parameters
    #     ----------
    #     gpis: iterable, optional
    #         if given these gpis will be used, can be practical
    #         if the gpis are managed by an external class e.g. for parallel
    #         processing
    #     tsaggkw: dict
    #         Keywords to give to the time series aggregation function
    #
    #
    #     Returns
    #     -------
    #     gpi_array: numpy.array
    #         numpy array of gpis in this batch
    #     ts_bulk: dict of numpy arrays
    #         for each variable one numpy array of shape
    #         (len(gpi_array), len(ts_aggregated))
    #     """
    #     # have to use the grid iteration as long as iter_ts only returns
    #     # data frame and no time series object including relevant metadata
    #     # of the time series
    #     i = 0
    #     gpi_bulk = []
    #     ts_bulk = {}
    #     ts_index = None
    #     if gpis is None:
    #         # get grid points can return either 3 or 4 values
    #         # depending on the grid type, gpis is the first in both cases
    #         gpi_info = list(self.tsreader.grid.grid_points())
    #         gpis = np.array(gpi_info[0], dtype=int)
    #     for gpi in gpis:
    #         gpi_bulk.append(gpi)
    #         ts = self.tsreader.read_ts(gpi)
    #         ts_agg = self.agg_func(ts, **tsaggkw)
    #         for column in ts_agg.columns:
    #             try:
    #                 ts_bulk[column].append(ts_agg[column].values)
    #             except KeyError:
    #                 ts_bulk[column] = []
    #                 ts_bulk[column].append(ts_agg[column].values)
    #
    #         if ts_index is None:
    #             ts_index = ts_agg.index
    #
    #         i += 1
    #         if i >= self.ts_buffer:
    #             for key in ts_bulk:
    #                 ts_bulk[key] = np.vstack(ts_bulk[key])
    #             gpi_array = np.hstack(gpi_bulk)
    #             yield gpi_array, ts_bulk
    #             ts_bulk = {}
    #             gpi_bulk = []
    #             i = 0
    #     if i > 0:
    #         for key in ts_bulk:
    #             ts_bulk[key] = np.vstack(ts_bulk[key])
    #         gpi_array = np.hstack(gpi_bulk)
    #         yield gpi_array, ts_bulk


if __name__ == '__main__':
    pass
