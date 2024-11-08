from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Table

from reboost.hpge.utils import _merge_arrays


def test_merge():
    
    ak_obj  =ak.Array({"evtid":[1,1,1,1,2,2,3],"edep":[100,50,1000,20,100,200,10]})
    bufer_rows = ak.Array({"evtid":[1,1],"edep":[60,50]})


    # should only remove te last element
    merged_idx_0, buffer_0,mode= _merge_arrays(ak_obj,None,0,100 ,True)

    assert ak.all(merged_idx_0.evtid ==[1,1,1,1,2,2])
    assert ak.all(merged_idx_0.edep ==[100,50,1000,20,100,200] )

    assert ak.all(buffer_0.evtid ==[3])
    assert ak.all(buffer_0.edep ==[10] )

    # delete input file
    assert mode=="of"

    # if delete input is false it should be appended
    _,_,mode= _merge_arrays(ak_obj,None,0,100 ,False)
    assert mode=="append"

    # now if idx isnt 0 or the max_idx should add the buffer and remove the end

    merged_idx, buffer,mode= _merge_arrays(ak_obj,bufer_rows,2,100 ,True)

    assert ak.all(merged_idx.evtid ==[1,1,1,1,1,1,2,2])
    assert ak.all(merged_idx.edep ==[60,50,100,50,1000,20,100,200] )

    assert ak.all(buffer.evtid ==[3])
    assert ak.all(buffer.edep ==[10] )

    assert mode =="append"

    # now for the final index just adds the buffer

    merged_idx_end, buffer_end,mode= _merge_arrays(ak_obj,bufer_rows,100,100 ,True)

    assert ak.all(merged_idx_end.evtid ==[1,1,1,1,1,1,2,2,3])
    assert ak.all(merged_idx_end.edep ==[60,50,100,50,1000,20,100,200,10] )

    assert buffer_end is None