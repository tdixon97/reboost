from __future__ import annotations

import logging

from reboost.hpge import processors, utils

log = logging.getLogger(__name__)


def build_hit(
    lh5_in_file: str, lh5_out_file: str, config: dict, buffer_len: int = int(5e6)
) -> None:
    for idx, d in enumerate(config.keys()):
        msg = f"...running event grouping for {d}"
        log.debug(msg)
        delete_input = bool(idx == 0)

        # build the processing chain
        # TODO replace this with a config file similar to pygama.build_evt

        proc = processors.def_chain(
            [
                processors.sort_data,
                processors.group_by_time,
                processors.sum_energy,
                processors.smear_energy,
            ],
            [{}, {"window": 10}, {}, {"reso": config[d]["reso"], "energy_name": "sum_energy"}],
        )

        utils.read_write_incremental(
            lh5_out_file,
            f"hit/{d}",
            proc,
            f"hit/{d}",
            lh5_in_file,
            buffer_len,
            delete_input=delete_input,
        )
