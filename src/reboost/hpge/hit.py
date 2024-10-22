from __future__ import annotations

import logging
from collections.abc import Iterable

from reboost.hpge import processors, utils

log = logging.getLogger(__name__)


def build_hit(
    lh5_in_file: str, lh5_out_file: str, detectors: Iterable[str | int], buffer_len: int = int(5e6)
) -> None:
    # build the processing chain
    proc = processors.def_chain(
        [processors.group_by_time, processors.sum_energy, processors.smear_energy],
        [{"window": 10}, {}, {"reso": 2, "energy_name": "summed_energy"}],
    )

    for idx, d in enumerate(detectors):
        msg = f"...running event grouping for {d}"
        log.debug(msg)
        delete_input = bool(idx == 0)
        utils.read_write_incremental(
            lh5_out_file,
            f"hit/{d}",
            proc,
            f"hit/{d}",
            lh5_in_file,
            buffer_len,
            delete_input=delete_input,
        )
