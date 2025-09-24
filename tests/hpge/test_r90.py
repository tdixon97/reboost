from __future__ import annotations

import awkward as ak

from reboost.hpge.psd import r90


def test_r90():
    elect_edep = [46.4, 150, 281, 58.6, 13.3]
    elect_xloc = [0.0497, 0.0498, 0.0499, 0.0498, 0.0498]
    elect_yloc = [-0.000126, -0.000124, -0.000166, -0.000234, -0.000268]
    elect_zloc = [-0.0195, -0.0194, -0.0193, -0.0194, -0.0195]

    gamma_edep = [
        0.00331,
        0.00650,
        0.02561,
        0.00926,
        29.86008,
        126.68419,
        35.51880,
        82.07933,
        87.33795,
        72.08262,
        81.74408,
        41.64307,
        34.90099,
        22.11428,
        6.37953,
        38.67890,
        29.49228,
        37.60893,
        35.02997,
    ]
    gamma_xloc = [
        0.02683,
        0.04228,
        0.04228,
        0.04228,
        0.04228,
        0.04233,
        0.04240,
        0.04238,
        0.04240,
        0.04243,
        0.04244,
        0.04245,
        0.04245,
        0.04246,
        0.03832,
        0.03833,
        0.03833,
        0.03834,
        0.03834,
    ]
    gamma_yloc = [
        0.02882,
        0.01661,
        0.01661,
        0.01661,
        0.01661,
        0.01659,
        0.01657,
        0.01658,
        0.01663,
        0.01666,
        0.01668,
        0.01667,
        0.01668,
        0.01668,
        0.00056,
        0.00056,
        0.00055,
        0.00054,
        0.00054,
    ]
    gamma_zloc = [
        -0.01340,
        -0.02097,
        -0.02097,
        -0.02097,
        -0.02097,
        -0.02100,
        -0.02104,
        -0.02098,
        -0.02096,
        -0.02095,
        -0.02094,
        -0.02093,
        -0.02093,
        -0.02093,
        -0.01069,
        -0.01069,
        -0.01069,
        -0.01068,
        -0.01068,
    ]

    edep = [elect_edep, gamma_edep]
    xloc = [elect_xloc, gamma_xloc]
    yloc = [elect_yloc, gamma_yloc]
    zloc = [elect_zloc, gamma_zloc]

    data = ak.Array(
        {
            "edep": edep,
            "xloc": xloc,
            "yloc": yloc,
            "zloc": zloc,
        }
    )

    r90_output = r90(data.edep, data.xloc, data.yloc, data.zloc)

    # Targets
    elect_r90 = 0.0001816
    gamma_r90 = 0.0157370

    assert round(r90_output[0], 7) == elect_r90
    assert round(r90_output[1], 7) == gamma_r90

    edep = [gamma_edep]
    xloc = [gamma_xloc]
    yloc = [gamma_yloc]
    zloc = [gamma_zloc]

    data = ak.Array(
        {
            "edep": edep,
            "xloc": xloc,
            "yloc": yloc,
            "zloc": zloc,
        }
    )

    r90_output = r90(data.edep, data.xloc, data.yloc, data.zloc)
    assert round(r90_output[0], 7) == gamma_r90
