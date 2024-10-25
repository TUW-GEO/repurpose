import os
import tempfile
import pytest
from pathlib import Path
import yaml

from repurpose.misc import deprecated, find_first_at_depth, delete_empty_directories

def test_deprecated_function():
    @deprecated()
    @deprecated("My message")
    def oldfunc():
        pass

    with pytest.deprecated_call():
        oldfunc()

def test_find_first_at_depth():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        os.mkdir(tempdir / "level1")
        os.mkdir(tempdir / "level1" / "level2")

        with open(tempdir / "level1" / "level2" / "test.yml", 'w') as f:
            yaml.dump({'test': 1}, f, default_flow_style=False)

        assert find_first_at_depth(tempdir, 0) == 'level1'
        assert find_first_at_depth(tempdir, 1) == 'level2'
        assert find_first_at_depth(tempdir, 2) == 'test.yml'

        os.remove(tempdir / "level1" / "level2" / "test.yml")

        delete_empty_directories(tempdir / "level1")

        assert find_first_at_depth(tempdir, 0) == 'level1'
        assert find_first_at_depth(tempdir, 1) is None
