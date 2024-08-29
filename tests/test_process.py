import numpy as np
import pandas as pd
import time
import tempfile
import logging
import pytest

from repurpose.process import parallel_process, idx_chunks

def test_index_chunks():
    timestamps = pd.date_range('2020-07-01', '2020-07-09', freq='1D')
    for i, chunk in enumerate(idx_chunks(timestamps, n=3)):
        if i == 0:
            assert np.all(chunk == timestamps[:3])
        elif i == 1:
            assert np.all(chunk == timestamps[3:6])
        elif i == 2:
            assert np.all(chunk == timestamps[6:])
        else:
            raise AssertionError('More chunks than expected!')


def func(x: int, p: int):
    time.sleep(0.5)
    logging.info(f'x={x}, p={p}')
    return x**p


@pytest.mark.parametrize("n_proc,backend", [
    ("1", None),  # backend doesn't matter in this case
    ("2", "threading"), ("2", "multiprocessing"), ("2", "loky")
    ])
def test_apply_to_elements(n_proc, backend):
    iter_kwargs = {'x': [1, 2, 3, 4]}
    static_kwargs = {'p': 2}
    with tempfile.TemporaryDirectory() as log_path:
        res = parallel_process(
            func, iter_kwargs, static_kwargs, n_proc=int(n_proc),
            show_progress_bars=False, verbose=False, loglevel="DEBUG",
            ignore_errors=True, log_path=log_path, backend=backend)
        assert res == [1, 4, 9, 16]
