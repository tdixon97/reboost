from __future__ import annotations

import awkward as ak
import pytest
from dbetto import AttrsDict
from lgdo import Array, Struct, Table, VectorOfVectors, lh5

from reboost.build_evt import build_evt


@pytest.fixture(scope="module")
def test_gen_lh5(tmptestdir):
    # write a basic lh5 file

    hit_path = str(tmptestdir / "basic_hit.lh5")

    data_ch1 = {}
    data_ch1["energy"] = Array([100, 200, 300])
    tab1 = Table(data_ch1)

    data_ch2 = {}
    data_ch2["energy"] = Array([2615, 2042, 100, 500])
    tab2 = Table(data_ch2)

    lh5.write(Struct({"det1": tab1}), "hit", hit_path, wo_mode="of")
    lh5.write(Struct({"det2": tab2}), "hit", hit_path, wo_mode="append_column")

    # now make a TCM

    channels = ak.Array([[0], [1], [1], [0, 1], [1, 0]])
    rows = ak.Array([[0], [0], [1], [1, 2], [3, 2]])

    tcm = Table(
        {"table_key": VectorOfVectors(channels), "row_in_table": VectorOfVectors(rows)},
        attrs={"tables": "['stp/det1','stp/det2']"},
    )

    return hit_path, tcm


def test_basic(test_gen_lh5):
    ch_groups = AttrsDict({"det1": "on", "det2": "off"})
    pars = AttrsDict(
        {
            "p01-r001": {"reso": {"det1": [1, 0.1], "det2": [2, 0.2]}},
            "p01-r002": {"reso": {"det1": [1, 0.2], "det2": [1, 0.2]}},
        }
    )
    run_part = AttrsDict({"p01-r001": 2, "p01-r002": 3})

    evts = build_evt(
        test_gen_lh5[1],
        hitfile=test_gen_lh5[0],
        outfile=None,
        channel_groups=ch_groups,
        pars=pars,
        run_part=run_part,
    )
    assert isinstance(evts, Table)
    print(evts)
