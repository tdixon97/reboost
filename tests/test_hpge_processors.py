from __future__ import annotations

import numpy as np

from reboost.hpge import processors


def test_smear():
    truth = np.array([1, 2, 3, 4, 5])
    smeared = processors.smear_energies(truth, reso=0.01)

    assert np.size(smeared) == 5
