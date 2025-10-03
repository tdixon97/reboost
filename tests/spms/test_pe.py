from __future__ import annotations

import awkward as ak

from reboost.spms.pe import corrected_photoelectrons


def test_forced_trigger_correction():
    # check that with every data event empty it does nothing

    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]), ak.Array([[], [0], [0, 1]]), ak.Array([[]]), ak.Array([[]])
    )

    assert ak.all(pe == [[], [1], [2, 3]])
    assert ak.all(uid == [[], [0], [0, 1]])

    # check adding a constant
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]), ak.Array([[], [0], [0, 1]]), ak.Array([[1]]), ak.Array([[0]])
    )

    assert ak.all(pe == [[1], [2], [3, 3]])
    assert ak.all(uid == [[0], [0], [0, 1]])

    # check sorting
    orig_pe = ak.Array([[], [1], [2, 3]])
    pe, uid = corrected_photoelectrons(
        orig_pe,
        ak.Array([[], [0], [0, 1]]),
        ak.Array([[1, 3]]),
        ak.Array([[1, 0]]),
    )
    assert ak.all(pe == [[3, 1], [4, 1], [5, 4]])
    assert ak.all(uid == [[0, 1], [0, 1], [0, 1]])

    # check original pe are not change
    assert ak.all(orig_pe == [[], [1], [2, 3]])

    # check summed pe - make each event 6 pe
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]),
        ak.Array([[], [0], [0, 1]]),
        ak.Array([[1, 2, 3], [6], [3, 3], [5, 1]]),
        ak.Array([[1, 0, 2], [1], [2, 1], [0, 2]]),
    )

    sum_pe = ak.sum(pe, axis=-1)
    assert ak.all(sum_pe == [6, 7, 11], axis=-1)

    # test without sorting
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]),
        ak.Array([[], [0], [0, 1]]),
        ak.Array([[1, 2, 2], [4], [3, 3], [5, 1], []]),
        ak.Array([[1, 0, 2], [1], [2, 1], [0, 2], []]),
        allow_data_reuse=False,
    )

    assert ak.all(pe == [[2, 1, 2], [1, 4], [2, 6, 3]])
