from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from lgdo import lh5
from matplotlib import colors, widgets
from numpy.typing import NDArray

from .create import list_optical_maps

log = logging.getLogger(__name__)


def _get_weights(viewdata: dict):
    rolled = np.array([viewdata["axis"]] + [i for i in [2, 1, 0] if i != viewdata["axis"]])
    weights = viewdata["weights"].transpose(*rolled)

    extent_axes = [i for i in np.flip(rolled) if i != viewdata["axis"]]
    extent = [viewdata["edges"][i] for i in extent_axes]
    extent_d = [(e[1] - e[0]) / 2 for e in extent]
    extent = [(e[0] - extent_d[i], e[-1] + extent_d[i]) for i, e in enumerate(extent)]

    labels = [["x", "y", "z"][i] + " [m]" for i in extent_axes]

    return weights, np.array(extent).flatten(), labels


def _slice_text(viewdata: dict) -> str:
    axis_name = ["x", "y", "z"][viewdata["axis"]]
    axis_pos = viewdata["edges"][viewdata["axis"]][viewdata["idx"]]
    return f"{viewdata['detid']} | slice {axis_name} = {axis_pos:.2f} m"


def _process_key(event) -> None:
    fig = event.canvas.figure
    viewdata = fig.__reboost

    if event.key == "up":
        viewdata["axis"] = min(viewdata["axis"] + 1, 2)
    elif event.key == "down":
        viewdata["axis"] = max(viewdata["axis"] - 1, 0)
    elif event.key == "right":
        viewdata["idx"] += 1
    elif event.key == "left":
        viewdata["idx"] -= 1
    elif event.key == "c":
        _channel_selector(fig)

    max_idx = viewdata["weights"].shape[viewdata["axis"]] - 1
    viewdata["idx"] = max(min(viewdata["idx"], max_idx), 0)

    _update_figure(fig)


def _update_figure(fig) -> None:
    viewdata = fig.__reboost
    w, extent, labels = _get_weights(viewdata)

    ax = fig.axes[0]
    ax.texts[0].set_text(_slice_text(viewdata))
    ax.images[0].set_array(w[viewdata["idx"]])
    ax.images[0].set_extent(extent)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_anchor("C")
    fig.canvas.draw()


def _channel_selector(fig) -> None:
    axbox = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    channels = fig.__reboost["available_dets"]
    tb = widgets.RadioButtons(axbox, channels, active=channels.index(fig.__reboost["detid"]))

    def change_detector(label: str | None) -> None:
        if fig.__reboost["detid"] != label:
            fig.__reboost["detid"] = label
            edges, weights, _, _ = _prepare_data(*fig.__reboost["prepare_args"], label)
            fig.__reboost["weights"] = weights
            fig.__reboost["edges"] = edges
        tb.disconnect_events()
        axbox.remove()
        _update_figure(fig)

    tb.on_clicked(change_detector)
    fig.canvas.draw()


def _read_data(
    optmap_fn: str,
    detid: str = "all",
    histogram_choice: str = "p_det",
) -> tuple[tuple[NDArray], NDArray]:
    histogram = histogram_choice if histogram_choice != "p_det_err_rel" else "p_det"
    optmap_all = lh5.read(f"/{detid}/{histogram}", optmap_fn)
    optmap_edges = tuple([b.edges for b in optmap_all.binning])
    optmap_weights = optmap_all.weights.nda.copy()
    if histogram_choice == "p_det_err_rel":
        optmap_err = lh5.read(f"/{detid}/p_det_err", optmap_fn)
        divmask = optmap_weights > 0
        optmap_weights[divmask] = optmap_err.weights.nda[divmask] / optmap_weights[divmask]
        optmap_weights[~divmask] = -1

    return optmap_edges, optmap_weights


def _prepare_data(
    optmap_fn: str,
    divide_fn: str | None = None,
    cmap_min: float | Literal["auto"] = 1e-4,
    cmap_max: float | Literal["auto"] = 1e-2,
    histogram_choice: str = "p_det",
    detid: str = "all",
) -> tuple[tuple[NDArray], NDArray]:
    optmap_edges, optmap_weights = _read_data(optmap_fn, detid, histogram_choice)

    if divide_fn is not None:
        divide_edges, divide_map = _read_data(divide_fn, detid, histogram_choice)
        divmask = divide_map > 0
        optmap_weights[divmask] = optmap_weights[divmask] / divide_map[divmask]
        optmap_weights[~divmask] = -1

    if cmap_min == "auto":
        cmap_min = max(1e-10, optmap_weights[optmap_weights > 0].min())
    if cmap_max == "auto":
        cmap_max = optmap_weights.max()

    lower_count = np.sum((optmap_weights > 0) & (optmap_weights < cmap_min))
    if lower_count > 0:
        log.warning(
            "%d cells are non-zero and lower than the current colorbar minimum %.2e",
            lower_count,
            cmap_min,
        )
    higher_count = np.sum(optmap_weights > cmap_max)
    if higher_count > 0:
        log.warning(
            "%d cells are non-zero and higher than the current colorbar maximum %.2e",
            higher_count,
            cmap_max,
        )

    # set zero/close-to-zero values to a very small, but nonzero value. This means we can
    # style those cells using the `under` style and, can style `bad` (i.e. everything < 0
    # after this re-assignment) in a different way.
    optmap_weights[(optmap_weights >= 0) & (optmap_weights < 1e-50)] = min(1e-10, cmap_min)

    return optmap_edges, optmap_weights, cmap_min, cmap_max


def view_optmap(
    optmap_fn: list[str],
    detid: str = "all",
    divide_fn: str | None = None,
    start_axis: int = 2,
    cmap_min: float | Literal["auto"] = 1e-4,
    cmap_max: float | Literal["auto"] = 1e-2,
    histogram_choice: str = "p_det",
    title: str | None = None,
) -> None:
    available_dets = list_optical_maps(optmap_fn)

    prepare_args = (optmap_fn, divide_fn, cmap_min, cmap_max, histogram_choice)
    edges, weights, cmap_min, cmap_max = _prepare_data(*prepare_args, detid)

    fig = plt.figure(figsize=(10, 10))
    fig.canvas.mpl_connect("key_press_event", _process_key)
    start_axis_len = edges[start_axis].shape[0] - 1
    fig.__reboost = {
        "axis": start_axis,
        "weights": weights,
        "detid": detid,
        "edges": edges,
        "idx": min(int(start_axis_len / 2), start_axis_len - 1),
        "available_dets": available_dets,
        "prepare_args": prepare_args,
    }

    cmap = plt.cm.plasma.with_extremes(bad="w", under="gray", over="red")
    weights, extent, labels = _get_weights(fig.__reboost)
    plt.imshow(
        weights[fig.__reboost["idx"]],
        norm=colors.LogNorm(vmin=cmap_min, vmax=cmap_max),
        aspect=1,
        interpolation="none",
        cmap=cmap,
        extent=extent,
        origin="lower",
    )

    if title is None:
        title = Path(optmap_fn).stem
        if divide_fn is not None:
            title += " / " + Path(divide_fn).stem

    plt.suptitle(title)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.text(0, 1.02, _slice_text(fig.__reboost), transform=fig.axes[0].transAxes)
    plt.colorbar()
    plt.show()
