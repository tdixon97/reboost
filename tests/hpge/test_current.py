from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors

from reboost.hpge import psd, surface
from reboost.shape import cluster


@pytest.fixture(scope="module")
def test_model():
    # test getting the model
    model, x = psd.get_current_template(
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

    mu = -x[np.argmax(model)]

    # with fixed mu
    model, x = psd.get_current_template(
        -1000,
        3000,
        1.0,
        amax=1,
        mean_aoe=0.5,
        mu=mu,
        sigma=100,
        tau=100,
        tail_fraction=0.65,
        high_tail_fraction=0.1,
        high_tau=10,
    )

    return model, x


def test_maximum_current(test_model):
    model, x = test_model

    edep = VectorOfVectors(
        ak.Array([[100.0, 300.0, 50.0], [10.0, 0.0, 100.0], [500.0]]), attrs={"unit": "keV"}
    )
    times = VectorOfVectors(
        ak.Array([[400, 500, 700], [800, 0, 1500], [700]], attrs={"unit": "ns"})
    )

    curr = psd.maximum_current(edep, times, template=model, times=x)
    assert isinstance(curr, Array)

    assert len(curr) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(curr[2] - 250) < 0.1

    # test other return modes
    max_t = psd.maximum_current(
        edep,
        times,
        template=model,
        times=x,
        return_mode="max_time",
    )

    assert isinstance(max_t, Array)
    assert len(max_t) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(max_t[2] - 700) < 2

    energy = psd.maximum_current(
        edep,
        times,
        template=model,
        times=x,
        return_mode="energy",
    )

    assert isinstance(energy, Array)
    assert len(energy) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(energy[2] - 500.0) < 2


def test_with_cluster(test_model):
    model, x = test_model

    edep = VectorOfVectors(
        ak.Array([[100.0, 300.0, 50.0], [10.0, 1.0, 100.0], [500.0]]), attrs={"unit": "keV"}
    )
    times = VectorOfVectors(
        ak.Array([[400, 410, 420], [800, 0, 1500], [700]], attrs={"unit": "ns"})
    )
    xloc = VectorOfVectors(ak.Array([[1, 1.1, 1.2], [0, 50, 80], [100]], attrs={"unit": "mm"}))

    dist = VectorOfVectors(ak.Array([[50, 40, 0.2], [300, 0.4, 0.2], [0.8]], attrs={"unit": "ns"}))

    yloc = ak.full_like(xloc, 0.0)
    zloc = ak.full_like(xloc, 0.0)
    trackid = ak.full_like(xloc, 0)

    clusters = cluster.cluster_by_step_length(
        trackid, xloc, yloc, zloc, dist, threshold=1, threshold_surf=1, surf_cut=0
    )
    cluster_edep = cluster.apply_cluster(clusters, edep).view_as("ak")
    cluster_times = cluster.apply_cluster(clusters, times).view_as("ak")

    e = ak.sum(cluster_edep, axis=-1)
    t = ak.sum(cluster_edep * cluster_times, axis=-1) / e
    curr = psd.maximum_current(e, t, template=model, times=x)

    assert isinstance(curr, Array)
    assert len(curr) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(curr[0] - 225) < 0.1


def test_maximum_current_surface(test_model):
    model, x = test_model

    # test for both input types
    for dtype in [np.float64, np.float32]:
        edep = VectorOfVectors(
            ak.values_astype(ak.Array([[100.0, 300.0, 50.0], [10.0, 0.0, 100.0], [500.0]]), dtype),
            attrs={"unit": "keV"},
        )

        times = VectorOfVectors(
            ak.values_astype(
                ak.Array([[400, 500, 700], [800, 0, 1500], [700]], attrs={"unit": "ns"}), dtype
            )
        )

        dist = VectorOfVectors(
            ak.values_astype(
                ak.Array([[50, 40, 0.2], [300, 0.4, 0.2], [0.8]], attrs={"unit": "ns"}), dtype
            )
        )

        surface_models = surface.get_surface_library(1002, 10)

        assert np.shape(surface_models)[0] == 10000
        assert np.shape(surface_models)[1] == 100
        surface_templates = psd.make_convolved_surface_library(model, surface_models)
        surface_activeness = surface_models[:, -1]

        curr_surf = psd.maximum_current(
            edep,
            times,
            dist,
            template=model,
            fccd_in_um=1002,
            templates_surface=surface_templates,
            activeness_surface=surface_activeness,
            times=x,
            return_mode="current",
        ).view_as("np")

        curr_bulk = psd.maximum_current(
            edep,
            times,
            dist,
            template=model,
            times=x,
            return_mode="current",
        ).view_as("np")
        # check shape

        assert len(curr_surf) == 3

        # surface effects reduce the current
        assert np.all(curr_surf < curr_bulk)
