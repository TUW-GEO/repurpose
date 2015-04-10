import numpy as np

def agg_tsmonthly(ts, **kwargs):
    """
    Parameters
    ----------
    ts : pandas.DataFrame
        time series of a point
    kwargs : dict
        any additional keyword arguments that are given to the ts2img object
        during initialization

    Returns
    -------
    ts_agg : pandas.DataFrame
        aggregated time series, they all must have the same length
        otherwise it can not work
        each column of this DataFrame will be a layer in the image
    """
    # very simple example
    # aggregate to monthly timestamp
    # should also make sure that the output has a certain length
    return ts.asfreq("M")

class Ts2Img(object):

    """
    Takes a time series dataset and converts it
    into an image dataset.
    A custom aggregate function should be given otherwise
    a daily mean will be used

    Parameters
    ----------
    tsreader: object
        object that implements a iter_ts method which iterates over
        pandas time series and has a grid attribute that is a pytesmo
        BasicGrid or CellGrid
    imgwriter: object
        writer object that implements a write_ts method that takes
        a list of grid point indices and a 2D array containing the time series data
    startdate: type
        description
    enddate: type
        description
    agg_func: function
        function that takes a pandas DataFrame and returns
        an aggregated pandas DataFrame
    ts_buffer: int
        how many time series to read before writing to disk,
        constrained by the working memory the process should use.

    """

    def __init__(self, tsreader, imgwriter,
                 startdate, enddate, agg_func=None,
                 ts_buffer=1000):

        self.agg_func = agg_func
        if self.agg_func is None:
            try:
                self.agg_func = tsreader.agg_ts2img
            except AttributeError:
                self.agg_func = agg_ts2img
        self.tsreader = tsreader
        self.imgwriter = imgwriter
        self.startdate = startdate
        self.enddate = enddate
        self.ts_buffer = ts_buffer

    def calc(self, **tsaggkw):
        """
        does the conversion from time series to images
        """
        for gpis, ts in self.tsbulk(**tsaggkw):
            self.imgwriter.write_ts(gpis, ts)

    def tsbulk(self, gpis=None, **tsaggkw):
        """
        iterator over gpi and time series arrays of size self.ts_buffer

        Parameters
        ----------
        gpis: iterable, optional
            if given these gpis will be used, can be practical
            if the gpis are managed by an external class e.g. for parallel
            processing
        tsaggkw: dict
            Keywords to give to the time series aggregation function


        Returns
        -------
        gpi_array: numpy.array
            numpy array of gpis in this batch
        ts_bulk: dict of numpy arrays
            for each variable one numpy array of shape
            (len(gpi_array), len(ts_aggregated))
        """
        # have to use the grid iteration as long as iter_ts only returns
        # data frame and no time series object including relevant metadata
        # of the time series
        i = 0
        gpi_bulk = []
        ts_bulk = {}
        ts_index = None
        if gpis is None:
            gpis, _, _, _ = self.tsreader.grid.grid_points()
        for gpi in gpis:
            gpi_bulk.append(gpi)
            ts = self.tsreader.read_ts(gpi)
            ts_agg = self.agg_func(ts, **tsaggkw)
            for column in ts_agg.columns:
                try:
                    ts_bulk[column].append(ts_agg[column].values)
                except KeyError:
                    ts_bulk[column] = []
                    ts_bulk[column].append(ts_agg[column].values)

            if ts_index is None:
                ts_index = ts_agg.index

            i += 1
            if i >= self.ts_buffer:
                for key in ts_bulk:
                    ts_bulk[key] = np.hstack(ts_bulk[key])
                gpi_array = np.hstack(gpi_bulk)
                yield gpi_array, ts_bulk
                ts_bulk = {}
                gpi_bulk = []
                i = 0
