import time
import os

# Note: Must be set BEFORE the first numpy import!!
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import traceback
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import sys
from pathlib import Path
from typing import List, Any
from glob import glob
from joblib import Parallel, delayed, parallel_config
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager


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
    p = str(os.path.join(os.path.dirname(os.path.abspath(__file__))))
    return Path(p).parents[1]


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

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, desc="",
                 *args, **kwargs) -> None:
        """
        Joblib parallel with progress bar
        """
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Wraps progress bar around function calls
        """
        with tqdm(
            disable=not self._use_tqdm, total=self._total, desc=self._desc
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)
    def print_progress(self):
        """
        Updated the progress bar after each successful call
        """
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def configure_worker_logger(log_queue, log_level, name):
    worker_logger = logging.getLogger(name)
    if not worker_logger.hasHandlers():
        h = QueueHandler(log_queue)
        worker_logger.addHandler(h)
    worker_logger.setLevel(log_level)
    return worker_logger

def run_with_error_handling(FUNC,
                            ignore_errors=False,
                            log_queue=None,
                            log_level="WARNING",
                            logger_name=None,
                            **kwargs) -> Any:

    if log_queue is not None:
        logger = configure_worker_logger(log_queue, log_level, logger_name)
    else:
        # normal logger
        logger = logging.getLogger(logger_name)

    r = None

    try:
        r = FUNC(**kwargs)
    except Exception as e:
        if ignore_errors:
            logger.error(f"The following ERROR was raised in the parallelized "
                         f"function `{FUNC.__name__}` but was ignored due to "
                         f"the chosen settings: "
                         f"{traceback.format_exc()}")
        else:
            raise e
    return r

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
        logger_name=None,
        verbose=False,
        progress_bar_label="Processed",
        backend="loky",
        sharedmem=False,
        parallel_kwargs=None,
) -> list:
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
        Which level should be logged. Must be one of
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].
    logger_name: str, optional (default: None)
        The name to assign to the logger that can be accessed in FUNC to
        log to. If not given, then the root logger is used. e.g
        ```
        logger = logging.getLogger(<logger_name>)
        logger.error("Some error message")
        ```
    verbose: bool, optional (default: False)
        Print all logging messages to stdout, useful for debugging.
        Only effective when logging is activated.
    progress_bar_label: str, optional (default: "Processed")
        Label to use for the progress bar.
    backend: Literal["threading", "multiprocessing", "loky"] = "loky"
        The backend to use for parallel execution (if n_proc > 1).
        Defaults to "loky". See joblib docs for more info.
    sharedmem: bool, optional (default:True)
        Activate shared memory option (slow)

    Returns
    -------
    results: list or None
        List of return values from each function call or None if no return
        values are found.
    """
    if STATIC_KWARGS is None:
        STATIC_KWARGS = dict()

    if activate_logging:
        logger = logging.getLogger(logger_name)
        logger.setLevel(loglevel.upper())

        if verbose:
            # in this case we also print ALL log messages
            streamHandler = logging.StreamHandler(sys.stdout)
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
            # in this case the logger should write to file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            filehandler = logging.FileHandler(log_file)
            filehandler.setFormatter(logging.Formatter(
                "%(levelname)s %(asctime)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(filehandler)
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

    if n_proc == 1:
        results = []
        if show_progress_bars:
            pbar = tqdm(total=len(process_kwargs), desc=progress_bar_label)
        else:
            pbar = None

        for kwargs in process_kwargs:
            r = run_with_error_handling(FUNC, ignore_errors,
                                        logger_name=logger_name,
                                        **kwargs)
            if r is not None:
                results.append(r)
            if pbar is not None:
                pbar.update()
    else:
        if logger is not None:
            log_level = logger.getEffectiveLevel()
            m = Manager()
            q = m.Queue()
            listener = QueueListener(q, *logger.handlers,
                                     respect_handler_level=True)
            listener.start()
        else:
            q = None
            log_level = None
            listener = None

        n = 1 if backend == 'loky' else None
        with parallel_config(backend=backend, inner_max_num_threads=n):
            results: list = ProgressParallel(
                use_tqdm=show_progress_bars,
                n_jobs=n_proc,
                verbose=0,
                total=len(process_kwargs),
                desc=progress_bar_label,
                require='sharedmem' if sharedmem else None,
                return_as="list",
                **parallel_kwargs or dict(),
            )(delayed(run_with_error_handling)(
                FUNC, ignore_errors,
                log_queue=q,
                log_level=log_level,
                logger_name=logger_name,
                **kwargs)
                for kwargs in process_kwargs)

            results = [r for r in results if r is not None]

        if listener is not None:
            listener.stop()

    if logger is not None:
        if verbose:
            logger.handlers.clear()

        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        handlers.clear()

    if len(results) == 0:
        return None
    else:
        return results
