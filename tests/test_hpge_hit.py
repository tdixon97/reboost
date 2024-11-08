from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Table

from reboost.hpge import hit


def test_eval():
    in_arr = ak.Array(
        {
            "evtid": [[1, 1, 1], [2, 2], [10, 10], [11], [12, 12, 12]],
            "time": [[0, 0, 0], [0, 0], [0, 0], [0], [0, 0, 0]],
            "edep": [[100, 150, 300], [0, 2000], [10, 20], [19], [100, 200, 300]],
        }
    )
    tab = Table(in_arr)
    basic_eval = {"mode": "eval", "expression": "ak.sum(hit.edep,axis=-1)"}

    assert ak.all(
        hit.eval_expression(tab, basic_eval, {}).view_as("ak") == [550, 2000, 30, 19, 600]
    )

    tab.add_field("e_sum", hit.eval_expression(tab, basic_eval, {}))
    bad_eval = {"mode": "sum", "expression": "ak.sum(hit.edep,axis=-1)"}

    with pytest.raises(ValueError):
        hit.eval_expression(tab, bad_eval, {})

    func_eval = {
        "mode": "function",
        "expression": "reboost.hpge.processors.smear_energies(hit.e_sum,reso=pars.reso)",
    }
    pars = {"reso": 2}

    assert np.size(hit.eval_expression(tab, func_eval, pars).view_as("np")) == 5
