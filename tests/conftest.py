from __future__ import annotations

import shutil
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import numba
import pytest
from legendtestdata import LegendTestData

_tmptestdir = Path(gettempdir()) / f"reboost-tests-{getuser()}-{uuid.uuid4()!s}"


@pytest.fixture(scope="session")
def tmptestdir_global():
    _tmptestdir.mkdir(exist_ok=False)
    return _tmptestdir


@pytest.fixture(scope="session")
def legendtestdata():
    ldata = LegendTestData()
    ldata.checkout("1d069f1")
    return ldata


@pytest.fixture(scope="module")
def tmptestdir(tmptestdir_global, request):
    p = tmptestdir_global / request.module.__name__
    p.mkdir(exist_ok=True)  # note: will be cleaned up globally.
    return p


def pytest_sessionfinish(exitstatus):
    if exitstatus == 0 and Path.exists(_tmptestdir):
        shutil.rmtree(_tmptestdir)


def patch_numba_for_tests():
    """Globally disable numba cache and enable bounds checking (for testing)."""
    njit_old = numba.njit

    def njit_patched(*args, **kwargs):
        kwargs.update({"cache": False, "boundscheck": True})
        return njit_old(*args, **kwargs)

    numba.njit = njit_patched


patch_numba_for_tests()
