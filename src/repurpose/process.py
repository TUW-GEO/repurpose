import sys
import time
import warnings
import os

if 'numpy' in sys.modules:
    warnings.warn("Numpy is already imported. Environment variables set in "
                  "repurpose.utils wont have any effect!")

# Note: Must be set BEFORE the first numpy import!!
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from tqdm import tqdm
import logging
from multiprocessing import Pool
from datetime import datetime
import sys
from pathlib import Path
from typing import List
from glob import glob


class ImageBaseConnection:
    """
    Wrapper for image reader that creates a list of all files in a root
    directory upon initialisation.
    When the reader tries to access a file but cannot find it, verify agains
    the previously created list. If the file should exist, repeat the reading
    assuming that due to some temporary issue the file is not accessible.

    This protects against processing gaps due to e.g. temporary network issues.
    """

    def __init__(self, reader, max_retries=99, retry_delay_s=1):
        """
        Parameters
        ----------
        reader: MultiTemporalImageBase
            Reader object for which the filelist is created
        max_retries: int, optional (default: 10)
            Number of retries when a file is in the filelist but reading
            fails.
        retry_delay_s: int, optional (default: 1)
            Number of seconds to wait after each failed retry.
        """
        self.reader = reader
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

        self.filelist = self._gen_filelist()

    @property
    def grid(self):
        return self.reader.grid

    def tstamps_for_daterange(self, *args, **kwargs):
        return self.reader.tstamps_for_daterange(*args, **kwargs)

    def _gen_filelist(self) -> list:
        flist = glob(os.path.join(self.reader.path, '**'), recursive=True)
        return flist

    def read(self, timestamp, **kwargs):
        retry = 0
        img = None
        error = None
        filename = None

        while (img is None) and (retry <= self.max_retries):
            try:
                if filename is None:
                    filename = self.reader._build_filename(timestamp)
                img = self.reader.read(timestamp, **kwargs)
            except Exception as e:
                logging.error(f"Error reading file (try {retry+1}) "
                              f"at {timestamp}: {e}. "
                              f"Trying again.")
                if filename is not None:
                    if filename not in self.filelist:
                        logging.error(
                            f"File at {timestamp} does not exist.")
                        break
                # else:
                img = None
                error = e
                time.sleep(self.retry_delay_s)

            retry += 1

        if img is None:
            logging.error(f"Reading file at {timestamp} failed after "
                          f"{retry} retries: {error}")
        else:
            logging.info(f"Success reading {filename} after {retry} "
                         f"tries.")
        return img


def rootdir() -> Path:
    return Path(os.path.join(os.path.dirname(
        os.path.abspath(__file__)))).parents[1]


def idx_chunks(idx, n=-1):
    """
    Yield successive n-sized chunks from list.

    Parameters
    ----------
    idx : pd.DateTimeIndex
        Time series index to split into parts
    n : int, optional (default: -1)
        Parts to split idx up into, -1 returns the full index.
    """
    if n == -1:
        yield idx
    else:
        for i in range(0, len(idx.values), n):
            yield idx[i:i + n]


def parallel_process_async(
        FUNC,
        ITER_KWARGS,
        STATIC_KWARGS=None,
        n_proc=1,
        show_progress_bars=True,
        ignore_errors=False,
        activate_logging=True,
        log_path=None,
        log_filename=None,
        loglevel="WARNING",
        verbose=False,
        progress_bar_label="Processed"
) -> List:
    """
    Applies the passed function to all elements of the passed iterables.
    Parallel function calls are processed ASYNCHRONOUSLY (ie order of
    return values might be different from order of passed iterables)!
    Usually the iterable is a list of cells, but it can also be a list of
    e.g. images etc.

    Parameters
    ----------
    FUNC: Callable
        Function to call.
    ITER_KWARGS: dict
        Container that holds iterables to split up and call in parallel with
        FUNC:
        Usually something like 'cell': [cells, ... ]
        If multiple, iterables MUST HAVE THE SAME LENGTH.
        We iterate through all iterables and pass them to FUNC as individual
        kwargs. i.e. FUNC is called N times, where N is the length of
        iterables passed in this dict. Can not be empty!
    STATIC_KWARGS: dict, optional (default: None)
        Kwargs that are passed to FUNC in addition to each element in
        ITER_KWARGS. Are the same for each call of FUNC!
    n_proc: int, optional (default: 1)
        Number of parallel workers. If 1 is chosen, we do not use a pool. In
        this case the return values are kept in order.
    show_progress_bars: bool, optional (default: True)
        Show how many iterables were processed already.
    ignore_errors: bool, optional (default: False)
        If True, exceptions are caught and logged. If False, exceptions are
        raised.
    activate_logging: bool, optional (default: True)
        If False, no logging is done at all (neither to file nor to stdout).
    log_path: str, optional (default: None)
        If provided, a log file is created in the passed directory.
    log_filename: str, optional (default: None)
        Name of the logfile in `log_path to create. If None is chosen, a name
        is created automatically. If `log_path is None, this has no effect.
    loglevel: str, optional (default: "WARNING")
        Log level to use for logging. Must be one of
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].
    verbose: float, optional (default: False)
        Print all logging messages to stdout, useful for debugging.
    progress_bar_label: str, optional (default: "Processed")
        Label to use for the progress bar.

    Returns
    -------
    results: List
        List of return values from each function call
    """
    if activate_logging:
        logger = logging.getLogger()

        if STATIC_KWARGS is None:
            STATIC_KWARGS = dict()

        if verbose:
            streamHandler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            streamHandler.setFormatter(formatter)
            logger.setLevel('DEBUG')
            logger.addHandler(streamHandler)

        if log_path is not None:
            if log_filename is None:
                d = datetime.now().strftime('%Y%m%d%H%M')
                log_filename = f"{FUNC.__name__}_{d}.log"
            log_file = os.path.join(log_path, log_filename)
        else:
            log_file = None

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.basicConfig(
                filename=log_file,
                level=loglevel.upper(),
                format="%(levelname)s %(asctime)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                force=True,
            )
    else:
        logger = None

    n = np.array([len(v) for k, v in ITER_KWARGS.items()])
    if len(n) == 0:
        raise ValueError("No ITER_KWARGS passed")
    if len(n) > 1:
        if not np.all(np.diff(n) == 0):
            raise ValueError(
                "Different number of elements found in ITER_KWARGS."
                f"All passed Iterable must have the same length."
                f"Got: {n}")
    n = n[0]

    i1d = np.intersect1d(
        np.array(list(ITER_KWARGS.keys())),
        np.array(list(STATIC_KWARGS.keys())))
    if len(i1d) > 0:
        raise ValueError("Got duplicate(s) in ITER_KWARGS and STATIC_KWARGS. "
                         f"Must be unique. Duplicates: {i1d}")

    process_kwargs = []
    for i in range(n):
        kws = {k: v[i] for k, v in ITER_KWARGS.items()}
        kws.update(STATIC_KWARGS)
        process_kwargs.append(kws)

    if show_progress_bars:
        pbar = tqdm(total=len(process_kwargs), desc=progress_bar_label)
    else:
        pbar = None

    results = []

    def update(r) -> None:
        if r is not None:
            results.append(r)
        if pbar is not None:
            pbar.update()

    def error(e) -> None:
        if logger is not None:
            logging.error(e)
        if not ignore_errors:
            raise e
        if pbar is not None:
            pbar.update()

    if n_proc == 1:
        for kwargs in process_kwargs:
            try:
                r = FUNC(**kwargs)
                update(r)
            except Exception as e:
                error(e)
    else:
        with Pool(n_proc) as pool:
            for kwds in process_kwargs:
                pool.apply_async(
                    FUNC,
                    kwds=kwds,
                    callback=update,
                    error_callback=error,
                )
            pool.close()
            pool.join()

    if pbar is not None:
        pbar.close()

    if logger is not None:
        if verbose:
            logger.handlers.clear()

        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        handlers.clear()

    return results
