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
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]),
        ak.Array([[], [0], [0, 1]]),
        ak.Array([[1, 3]]),
        ak.Array([[1, 0]]),
    )
    assert ak.all(pe == [[3, 1], [4, 1], [5, 4]])
    assert ak.all(uid == [[0, 1], [0, 1], [0, 1]])
