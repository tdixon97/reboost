from __future__ import annotations

import awkward as ak
import numpy as np
from lgdo import Table

from reboost.hpge import hit, processors


def test_evtid_group():
    in_arr_evtid = ak.Array(
        {"evtid": [1, 1, 1, 2, 2, 10, 10, 11, 12, 12, 12], "time": np.zeros(11)}
    )

    in_tab = Table(in_arr_evtid)

    out = processors.group_by_evtid(in_tab)
    out_ak = out.view_as("ak")
    assert ak.all(out_ak.evtid == [[1, 1, 1], [2, 2], [10, 10], [11], [12, 12, 12]])
    assert ak.all(out_ak.time == [[0, 0, 0], [0, 0], [0, 0], [0], [0, 0, 0]])

    # test the eval in build hit also

    out_eval = hit.step_group(
        in_tab,
        {
            "description": "group steps by time and evtid.",
            "expression": "reboost.hpge.processors.group_by_evtid(stp)",
        },
    )

    out_eval_ak = out_eval.view_as("ak")

    assert ak.all(out_ak.evtid == out_eval_ak.evtid)
    assert ak.all(out_ak.time == out_eval_ak.time)


def test_time_group():
    # time units are ns
    in_arr_evtid = ak.Array(
        {
            "evtid": [1, 1, 1, 2, 2, 2, 2, 2, 11, 12, 12, 12, 15, 15, 15, 15, 15],
            "time": [
                0,
                -2000,
                3000,
                0,
                100,
                1200,
                17000,
                17010,
                0,
                0,
                0,
                -5000,
                150,
                151,
                152,
                3000,
                3100,
            ],
        }
    )

    in_tab = Table(in_arr_evtid)

    # 1us =1000ns
    out = processors.group_by_time(in_tab, window=1)
    out_ak = out.view_as("ak")
    assert ak.all(
        out_ak.evtid
        == [[1], [1], [1], [2, 2], [2], [2, 2], [11], [12], [12, 12], [15, 15, 15], [15, 15]]
    )
    assert ak.all(
        out_ak.time
        == [
            [-2000],
            [0],
            [3000],
            [0, 100],
            [1200],
            [17000, 17010],
            [0],
            [-5000],
            [0, 0],
            [150, 151, 152],
            [3000, 3100],
        ]
    )