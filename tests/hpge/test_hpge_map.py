from __future__ import annotations

import numpy as np
import pytest
from lgdo import Array, Struct, lh5
from scipy.interpolate import RegularGridInterpolator

from reboost.hpge import psd
from reboost.hpge.utils import HPGeRZField, get_hpge_rz_field
from reboost.units import ureg as u


def test_read_hpge_map(legendtestdata):
    dt_map = get_hpge_rz_field(
        legendtestdata["lh5/hpge-drift-time-maps.lh5"],
        "V99000A",
        "drift_time",
        out_of_bounds_val=0,
    )

    assert isinstance(dt_map, HPGeRZField)

    assert dt_map.r_units == u.m
    assert dt_map.z_units == u.m
    assert dt_map.φ_units == u.ns

    assert isinstance(dt_map.φ, RegularGridInterpolator)

    with pytest.raises(ValueError):
        dt_map.φ((0, -1))

    assert dt_map.φ((0, 0)) == 0
    assert dt_map.φ([(0, 0.01), (0.03, 0.03)]) == pytest.approx([135, 695])


@pytest.fixture(scope="module")
def test_pulse_shape_library(tmptestdir):
    model, _ = psd.get_current_template(
        -1000,
        3000,
        1.0,
        amax=1,
        mean_aoe=1,
        mu=0,
        sigma=100,
        tau=100,
        tail_fraction=0.65,
        high_tail_fraction=0.1,
        high_tau=10,
    )

    # loop
    r = z = np.linspace(0, 100, 200)
    waveforms = np.zeros((200, 200, 4001))
    for i in range(200):
        for j in range(200):
            waveforms[i, j] = model

    res = Struct(
        {
            "r": Array(r, attrs={"units": "mm"}),
            "z": Array(z, attrs={"units": "mm"}),
            "waveforms": Array(waveforms, attrs={"units": ""}),
        }
    )
    lh5.write(res, "V01", f"{tmptestdir}/pulse_shape_lib.lh5")

    return f"{tmptestdir}/pulse_shape_lib.lh5"


def test_read_pulse_shape_library(test_pulse_shape_library):
    # check th reading works
    lib = get_hpge_rz_field(test_pulse_shape_library, "V01", "waveforms")
    assert isinstance(lib, HPGeRZField)

    assert len(lib.φ((10, 10))) == 4001
