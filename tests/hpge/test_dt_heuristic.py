from __future__ import annotations

import numpy as np
import pytest
from dbetto import AttrsDict
from lgdo import Array, Table, VectorOfVectors, lh5
from scipy.interpolate import RegularGridInterpolator

from reboost.hpge import psd, utils
from reboost.units import ureg as u

# data from remage sim in ./simulation
# fmt: off
gamma_stp = Table(
    {
        "edep": VectorOfVectors(
            [
                [763, 20.1, 30.2, 40.5, 36.5, 110],
                [0.0158, 526, 111],
                [1.4, 12.4, 19, 9.99, 18.9, 23.4, 12.5, 6.55, 25.3, 24.7, 9.23, 32.7, 31.4, 35.2, 54.1, 27.1],
                [592, 248, 161],
                [1.4, 48.2, 145, 134, 119, 48.2, 46.9, 42.2, 34.8, 23.5, 36.1, 31.3, 57.4, 25.2],
                [0.179, 31.5, 83.9, 52.3, 100, 29.9, 29.4, 41.7, 27.8, 25.3, 27.3, 25.2, 35, 12.8, 106, 124, 247],
                [530, 0.179, 7.97, 29.6, 86.1, 39.3, 42.2],
                [737, 73.4, 190],
                [728, 16.1, 256],
                [613, 0.179, 9.03, 101, 26.9, 22.9, 26.1, 31.5, 11.1, 5.56, 31.7, 27.2, 23, 33.7, 37.6],
            ],
            attrs={"units": "keV"},
        ),
        "xloc": VectorOfVectors(
            [
                [-0.0022, 0.0115, 0.0141, 0.0169, 0.0165, 0.0165],
                [0.0106, 0.0104, 0.0103],
                [0.0135, 0.0135, 0.0135, 0.0135, 0.0135, 0.0136, 0.0136, 0.0136, 0.0136, 0.0136, 0.0135, 0.0135, 0.0136, 0.0136, 0.0136, 0.0136],
                [0.0025, 0.00037, 0.00041],
                [0.00132, 0.00134, 0.00139, 0.00139, 0.00134, 0.00132, 0.00131, 0.0013, 0.00129, 0.00128, 0.00128, 0.00127, 0.00127, 0.00127],
                [0.0128, 0.0128, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.0129, 0.00937, 0.00243, 0.00247],
                [-0.0127, -0.0224, -0.0224, -0.0224, -0.0224, -0.0224, -0.0224],
                [0.0118, 0.00888, 0.0085],
                [-0.00369, -0.00343, 0.00682],
                [-0.0162, -0.0224, -0.0224, -0.0224, -0.0224, -0.0224, -0.0224, -0.0224, -0.022, -0.022, -0.022, -0.022, -0.022, -0.022, -0.022],
            ],
            attrs={"units": "m"},
        ),
        "yloc": VectorOfVectors(
            [
                [-0.00951, -0.0201, -0.0199, -0.014, -0.00924, -0.0092],
                [-0.0264, -0.0264, -0.0264],
                [0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0198, 0.0197, 0.0197, 0.0197, 0.0197, 0.0197, 0.0197],
                [0.00671, 0.00169, 0.00209],
                [-0.00184, -0.00185, -0.00191, -0.00198, -0.00201, -0.002, -0.002, -0.002, -0.00199, -0.00199, -0.00199, -0.00198, -0.00198, -0.00198],
                [-0.00227, -0.00227, -0.00226, -0.00226, -0.00228, -0.00231, -0.00231, -0.0023, -0.0023, -0.00231, -0.00232, -0.00232, -0.00232, -0.00232, -0.00517, -0.0189, -0.0184],
                [-0.0123, -0.0251, -0.0251, -0.0251, -0.0251, -0.0251, -0.0251],
                [0.00864, 0.00258, 0.00243],
                [-0.000536, -0.000655, -0.00369],
                [-0.00355, -0.0134, -0.0134, -0.0134, -0.0134, -0.0134, -0.0134, -0.0134, -0.0133, -0.0133, -0.0133, -0.0133, -0.0133, -0.0133, -0.0133],
            ],
            attrs={"units": "m"},
        ),
        "zloc": VectorOfVectors(
            [
                [0.0281, 0.0124, 0.0112, 0.00313, 0.0052, 0.00495],
                [0.029, 0.0292, 0.0293],
                [0.000129, 0.000135, 0.000144, 0.000141, 0.000133, 0.000124, 0.000114, 0.000107, 0.000105, 0.000107, 0.000113, 0.000126, 0.00014, 0.00015, 0.000153, 0.000153],
                [0.00692, 0.0114, 0.0114],
                [0.00356, 0.00359, 0.00364, 0.00369, 0.00373, 0.00377, 0.00379, 0.0038, 0.00379, 0.00379, 0.00377, 0.00377, 0.00377, 0.00376],
                [0.00099, 0.000999, 0.00101, 0.00103, 0.00103, 0.00102, 0.00104, 0.00105, 0.00105, 0.00105, 0.00105, 0.00105, 0.00105, 0.00105, 0.0141, 0.022, 0.0257],
                [0.0247, 0.0245, 0.0245, 0.0245, 0.0245, 0.0245, 0.0245],
                [0.0146, 0.0147, 0.0138],
                [0.0197, 0.0195, 0.0117],
                [0.0029, 0.00108, 0.00108, 0.00107, 0.00106, 0.00106, 0.00106, 0.00106, 0.00128, 0.00128, 0.00129, 0.0013, 0.0013, 0.0013, 0.0013],
            ],
            attrs={"units": "m"},
        ),
    }
)
# fmt: on


@pytest.fixture(scope="module")
def dt_map(legendtestdata):
    return utils.get_hpge_scalar_rz_field(
        legendtestdata["lh5/hpge-drift-time-maps.lh5"],
        "V99000A",
        "drift_time",
        out_of_bounds_val=0,
    )


def test_read_map_from_disk(dt_map):
    assert isinstance(dt_map.φ, RegularGridInterpolator)
    assert dt_map.φ_units == u.ns
    assert dt_map.r_units == u.m
    assert dt_map.z_units == u.m


@pytest.fixture(scope="module")
def dt_map_dummy(legendtestdata):
    data = lh5.read("V99000A", legendtestdata["lh5/hpge-drift-time-maps.lh5"])
    data = AttrsDict({k: data[k].view_as("np", with_units=True) for k in ("r", "z", "drift_time")})

    nan_idx = np.isnan(data.drift_time.m)

    dt_dummy_z = np.arange(0.1, 2, step=0.023)
    drift_time = np.tile(dt_dummy_z, 38).reshape((38, 83))

    drift_time[nan_idx] = np.nan

    assert drift_time.shape == (data.drift_time.m.shape)

    interpolator = RegularGridInterpolator(
        (data.r.m, data.z.m),
        drift_time,
    )

    return utils.HPGeScalarRZField(
        interpolator,
        data.r.u,
        data.z.u,
        u.us,
    )


def test_drift_time_dummy(dt_map_dummy):
    gamma_stp_shift = Table(
        {
            "edep": gamma_stp.edep,
            "xloc": VectorOfVectors(gamma_stp.xloc.view_as("ak") + 10, attrs={"units": "m"}),
            "yloc": VectorOfVectors(gamma_stp.yloc.view_as("ak") + 10, attrs={"units": "m"}),
            "zloc": VectorOfVectors(gamma_stp.zloc.view_as("ak") + 10, attrs={"units": "m"}),
        }
    )

    # compute all drift-times with the dummy map
    dt_values = psd.drift_time(
        gamma_stp_shift.xloc,
        gamma_stp_shift.yloc,
        gamma_stp_shift.zloc,
        dt_map_dummy,
        coord_offset=(
            10,
            10,
            10,
        )
        * u.m,
    )
    # turn into an Awkward array so we can index
    dt_arr = dt_values.view_as("ak")

    # helper to pull out expected from the RegularGridInterpolator
    def expected_dt(event, step):
        stp = gamma_stp.view_as("ak")
        x = stp.xloc[event][step]
        y = stp.yloc[event][step]
        z = stp.zloc[event][step]
        r = np.sqrt(x**2 + y**2)
        # φ returns a scalar when passed (r, z)
        return dt_map_dummy.φ((r, z))

    # --- first check: event 0, step 1 ---
    got01 = dt_arr[0][1]
    exp01 = expected_dt(0, 1)
    assert got01 == pytest.approx(exp01), f"0,1 → got {got01}, expected {exp01}"

    # --- second check: event 3, step 2 (for instance) ---
    got32 = dt_arr[3][2]
    exp32 = expected_dt(3, 2)
    assert got32 == pytest.approx(exp32), f"3,2 → got {got32}, expected {exp32}"


def test_drift_time(dt_map):
    dt_values = psd.drift_time(
        gamma_stp.xloc,
        gamma_stp.yloc,
        gamma_stp.zloc,
        dt_map,
    )
    assert isinstance(dt_values, VectorOfVectors)
    assert dt_values.ndim == 2
    assert dt_values.attrs["units"] == "ns"

    # test whether this works with non-LGDOs
    data = gamma_stp.view_as("ak")
    dt_values_nolgdo = psd.drift_time(
        data.xloc,  # units should match the dt map units -> meters
        data.yloc,
        data.zloc,
        dt_map,
    )
    assert isinstance(dt_values_nolgdo, VectorOfVectors)
    assert dt_values_nolgdo == dt_values


def test_drift_time_heuristics(dt_map):
    dt_values = psd.drift_time(
        gamma_stp.xloc,
        gamma_stp.yloc,
        gamma_stp.zloc,
        dt_map,
    )

    dt_heu = psd.drift_time_heuristic(
        dt_values,
        gamma_stp.edep,
    )

    assert isinstance(dt_heu, Array)
    assert dt_heu.attrs["units"] == "ns/keV"
